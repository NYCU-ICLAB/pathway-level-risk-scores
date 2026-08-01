#!/usr/bin/env python3
"""
gene_vs_pathway_headtohead.py
=============================
Fair head-to-head: gene-level vs pathway-level survival models.

Directly answers the committee question:
    "Is a gene-level model actually better than the parsimonious
     pathway-level compact score?"

Design principles (honest, could go either way):
  * Same fixed 8:2 train/test split (model_input/filter1_age_fixed + sample_id_mapping).
  * GENOMIC-ONLY primary comparison. The published compact pathway score uses
    NO clinical covariates, so the gene-level arms are also genomic-only here
    (giving gene-level a clinical boost would be unfair to pathway). A
    +clinical robustness variant is available via --with-clinical.
  * All arms evaluated on the SAME test set with the SAME metric functions:
        Harrell C-index, Uno C-IPCW, time-dependent AUC@24m, Accuracy@24m.
  * Model-selection (Coxnet lambda) by 5-fold CV C-index on train (prefer sparser).
  * Uncertainty by TEST-SET bootstrap (B resamples): 95% CI per metric, and
    paired Delta-C vs the deployed compact model with a two-sided bootstrap p.

Arms (per cancer):
  compact_fixed       published beta on the final K compact PW features (deployed model)
  compact_refit       same K PW features, CoxPH refit on train (best-case pathway)
  gene_restricted     gene-event Coxnet, genes = union of retained-pathway genes
  gene_unrestricted   gene-event Coxnet, genes = ALL recurrently-altered panel genes
  gene_capped_K       gene_restricted forced to ~K features (equal parsimony)

Run:
    python3 scripts/gene_vs_pathway_headtohead.py --out-dir figures_tables/gene_vs_pathway_headtohead
    python3 scripts/gene_vs_pathway_headtohead.py --cancers BRCA --boot 200   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines.utils import concordance_index

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sklearn.model_selection import train_test_split
import build_3flag_enhanced as B
from build_3flag_enhanced import parse_10pathway, CANCER_PATHWAYS, ALIAS_MAP

warnings.filterwarnings("ignore")

DATA_DIR = ROOT / "model_input" / "filter1_age_fixed"
ID_MAP_DIR = ROOT / "model_input" / "sample_id_mapping"
INPUT_DIR = ROOT / "build_3flag_inputs"
PATHWAY_XLSX = INPUT_DIR / "tcga_10_pathway.xlsx"
FORMULA = ROOT / "compact_score" / "compact_score_formula.yaml"
DECISION = ROOT / "ibcga_run_records" / "final_feature_selection_decision.csv"

CANCERS = ["BRCA", "LUAD", "PAAD", "PRAD", "CRC"]
CANCER_ALIAS = {"BRCA": "IDC"}  # decision-file / yaml key for BRCA
FIXED_COMMON = ["ecog_z", "age_z",
                "SAMPLE_TYPE_Local_Recurrence", "SAMPLE_TYPE_Metastasis",
                "GENE_PANEL_IMPACT341", "GENE_PANEL_IMPACT410", "GENE_PANEL_IMPACT505"]
FIXED_CRC_EXTRA = ["site_rectal", "site_colorectal_NOS"]

ALPHA_L1 = 0.5           # elastic-net mix for gene-level Coxnet
N_LAMBDA = 100
CV_FOLDS = 5
PREVALENCE = 0.01
TAU = 24.0               # months, evaluation horizon for IPCW / td-AUC / accuracy
SEED = 42


# ----------------------------------------------------------------------------- data
def reproduce_sids(cancer):
    """Recover the SAMPLE_ID for each model_input row by replaying build_cancer's
    deterministic stratified split. The model_input CSVs dropped SAMPLE_ID via
    reset_index, and the sample_id_mapping/ files are in a DIFFERENT order, so
    row-order joining against them scrambles the gene events. Replaying the exact
    train_test_split(random_state=42, stratify=Event_OS) restores 1:1 alignment
    (verified: PW binary features, Event_OS, OS_MONTHS all match rowwise)."""
    df = pd.read_csv(B.INTEG_GENOMICS[cancer])
    df["Hugo_Symbol"] = df["Hugo_Symbol"].replace(ALIAS_MAP)
    meta = B.build_meta(df)
    meta = meta[meta["OS_MONTHS"] > 0]
    sids = meta.index.tolist()
    tr_sids, te_sids = train_test_split(
        sids, test_size=0.2, random_state=42, stratify=meta.loc[sids, "Event_OS"])
    return tr_sids, te_sids


def load_split(cancer):
    cdir = cancer
    train = pd.read_csv(DATA_DIR / cdir / f"{cdir}_train.csv")
    test = pd.read_csv(DATA_DIR / cdir / f"{cdir}_test.csv")
    tr_ids, te_ids = reproduce_sids(cancer)
    assert len(tr_ids) == len(train) and len(te_ids) == len(test), (
        f"{cancer}: reproduced split ({len(tr_ids)}/{len(te_ids)}) != "
        f"model_input rows ({len(train)}/{len(test)}); alignment unsafe")
    return train, test, tr_ids, te_ids


def integrated_csv(cancer):
    fn = {"BRCA": "Breast_integrated_genomics.csv",
          "PRAD": "Prostate_integrated_genomics.csv",
          "LUAD": "LUAD_integrated_genomics_v3.csv",
          "CRC":  "Colorectal_integrated_genomics.csv",
          "PAAD": "Pancreatic_integrated_genomics.csv"}[cancer]
    return INPUT_DIR / fn


def build_gene_events(cancer, sample_ids, gene_set=None):
    """Per-(gene, alteration) binary indicators. gene_set=None -> all panel genes."""
    df = pd.read_csv(integrated_csv(cancer), low_memory=False)
    df["Hugo_Symbol"] = df["Hugo_Symbol"].replace(ALIAS_MAP)
    if gene_set is not None:
        df = df[df["Hugo_Symbol"].isin(gene_set)]
    df = df[df["SAMPLE_ID"].isin(sample_ids)]
    rows = []
    for sid, sub in df.groupby("SAMPLE_ID"):
        feats = {"SAMPLE_ID": sid}
        for _, r in sub.iterrows():
            g = r["Hugo_Symbol"]
            if pd.notna(r["MUT"]) and str(r["MUT"]).strip():
                feats[f"g_{g}_mut"] = 1
            if pd.notna(r["CNA"]):
                try:
                    v = float(r["CNA"])
                    if v >= 2:
                        feats[f"g_{g}_amp"] = 1
                    elif v <= -2:
                        feats[f"g_{g}_del"] = 1
                except (ValueError, TypeError):
                    pass
            if pd.notna(r["SV"]) and str(r["SV"]).strip():
                feats[f"g_{g}_sv"] = 1
        rows.append(feats)
    fd = pd.DataFrame(rows).set_index("SAMPLE_ID").fillna(0)
    return fd.reindex(sample_ids, fill_value=0)


def retained_gene_set(cancer):
    pw = parse_10pathway(str(PATHWAY_XLSX))
    gs = set()
    for p in CANCER_PATHWAYS[cancer]:
        gs |= pw[p]["all"]
    return gs


# ----------------------------------------------------------------------------- metrics
def make_surv(df):
    return Surv.from_arrays(event=df["Event_OS"].astype(bool).to_numpy(),
                            time=df["OS_MONTHS"].astype(float).to_numpy())


def acc_at_tau(t_tr, r_tr, t_test, e_test, r_test, tau=TAU):
    """Binary accuracy@tau. Label: died<=tau ->1; alive at tau ->0; censored<tau dropped.
    Predicted positive = risk above train median."""
    cutoff = np.median(r_tr)
    label, pred = [], []
    for t, e, r in zip(t_test, e_test, r_test):
        if e and t <= tau:
            lab = 1
        elif t >= tau:
            lab = 0
        else:
            continue
        label.append(lab)
        pred.append(int(r > cutoff))
    if not label:
        return float("nan")
    label, pred = np.array(label), np.array(pred)
    return float((label == pred).mean())


def compute_metrics(y_tr, df_tr, df_te, r_tr, r_te):
    """r_* = risk scores (higher = worse). Returns dict of test metrics."""
    y_te = make_surv(df_te)
    t_te = df_te["OS_MONTHS"].to_numpy(float)
    e_te = df_te["Event_OS"].astype(bool).to_numpy()
    # Harrell C (lifelines: pass -risk so higher risk -> shorter survival)
    c_harrell = float(concordance_index(t_te, -r_te, e_te))
    # Uno C-IPCW capped at tau within follow-up support
    tau = min(TAU, float(t_te[e_te].max()) if e_te.any() else TAU)
    try:
        c_ipcw = float(concordance_index_ipcw(y_tr, y_te, r_te, tau=tau)[0])
    except Exception:
        c_ipcw = float("nan")
    # td-AUC@24
    try:
        auc, _ = cumulative_dynamic_auc(y_tr, y_te, r_te, times=[min(TAU, tau)])
        td_auc = float(np.atleast_1d(auc)[0])
    except Exception:
        td_auc = float("nan")
    acc = acc_at_tau(df_tr["OS_MONTHS"].to_numpy(float), r_tr, t_te, e_te, r_te)
    return {"c_index": c_harrell, "c_ipcw": c_ipcw, "td_auc_24": td_auc, "acc_24": acc}


# ----------------------------------------------------------------------------- models
def coxnet_select(X_tr, y_tr, cap_k=None, cap_mode="densest"):
    """Fit Coxnet path, pick lambda by 5-fold CV C-index (prefer sparser).
    If cap_k given, restrict to lambdas whose n_features <= cap_k. cap_mode='densest'
    picks the densest eligible model; cap_mode='cv_best' picks the eligible model with the
    best 5-fold CV concordance (sparser preferred on ties) — a fairer matched-size comparator."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    path = CoxnetSurvivalAnalysis(l1_ratio=ALPHA_L1, n_alphas=N_LAMBDA, max_iter=100000)
    path.fit(X_tr, y_tr)
    lambdas = path.alphas_
    nf = []
    for lam in lambdas:
        m = CoxnetSurvivalAnalysis(l1_ratio=ALPHA_L1, alphas=[lam], max_iter=100000)
        m.fit(X_tr, y_tr)
        nf.append(int(np.sum(np.abs(m.coef_[:, 0]) > 1e-8)))
    nf = np.array(nf)

    if cap_k is not None:
        elig = np.where((nf <= cap_k) & (nf > 0))[0]
        if len(elig) == 0:
            elig = np.where(nf > 0)[0]
        if cap_mode == "cv_best":
            cvc = {}
            for i in elig:
                fold = []
                for tri, vai in kf.split(X_tr):
                    try:
                        m = CoxnetSurvivalAnalysis(l1_ratio=ALPHA_L1, alphas=[lambdas[i]], max_iter=100000)
                        m.fit(X_tr[tri], y_tr[tri]); fold.append(m.score(X_tr[vai], y_tr[vai]))
                    except Exception:
                        fold.append(0.5)
                cvc[i] = float(np.mean(fold))
            best = max(cvc.values())
            li = max((i for i in elig if abs(cvc[i] - best) < 1e-6), key=lambda i: lambdas[i])
        else:
            li = elig[np.argmax(nf[elig])]   # densest model at/under the cap
        best_lam = lambdas[li]
    else:
        cv = np.full(len(lambdas), np.nan)
        for i, lam in enumerate(lambdas):
            if nf[i] < 1:            # skip empty models: no discrimination, not a real "gene model"
                continue
            fold = []
            for tri, vai in kf.split(X_tr):
                try:
                    m = CoxnetSurvivalAnalysis(l1_ratio=ALPHA_L1, alphas=[lam], max_iter=100000)
                    m.fit(X_tr[tri], y_tr[tri])
                    fold.append(m.score(X_tr[vai], y_tr[vai]))
                except Exception:
                    fold.append(0.5)
            cv[i] = np.mean(fold)
        best = np.nanmax(cv)
        tied = np.where(np.abs(cv - best) < 1e-6)[0]
        li = tied[np.argmax(lambdas[tied])]  # prefer sparser (larger lambda) among CV-best
        best_lam = lambdas[li]

    final = CoxnetSurvivalAnalysis(l1_ratio=ALPHA_L1, alphas=[best_lam], max_iter=100000)
    final.fit(X_tr, y_tr)
    n_used = int(np.sum(np.abs(final.coef_[:, 0]) > 1e-8))
    return final, n_used


def run_gene_arm(cancer, train, test, tr_ids, te_ids, gene_set, cap_k=None, with_clinical=False):
    all_ids = tr_ids + te_ids
    feats = build_gene_events(cancer, all_ids, gene_set)
    prev = feats.loc[tr_ids].mean(0)
    feats = feats[prev[prev >= PREVALENCE].index]
    Xtr_g = feats.loc[tr_ids].reset_index(drop=True)
    Xte_g = feats.loc[te_ids].reset_index(drop=True)
    if with_clinical:
        fixed = [c for c in FIXED_COMMON if c in train.columns]
        if cancer == "CRC":
            fixed += [c for c in FIXED_CRC_EXTRA if c in train.columns]
        Xtr = pd.concat([train[fixed].reset_index(drop=True), Xtr_g], axis=1)
        Xte = pd.concat([test[fixed].reset_index(drop=True), Xte_g], axis=1)
    else:
        Xtr, Xte = Xtr_g, Xte_g
    scaler = StandardScaler().fit(Xtr.values)
    Xtr_s, Xte_s = scaler.transform(Xtr.values), scaler.transform(Xte.values)
    y_tr = make_surv(train)
    model, n_used = coxnet_select(Xtr_s, y_tr, cap_k=cap_k)
    r_tr = model.predict(Xtr_s)
    r_te = model.predict(Xte_s)
    return r_tr, r_te, n_used, Xtr.shape[1]


def run_compact_fixed(cancer, train, test, formula):
    from compact_score.apply_compact_score import apply_score  # noqa
    key = CANCER_ALIAS.get(cancer, cancer)
    _, r_tr = apply_score(train, key, formula)
    _, r_te = apply_score(test, key, formula)
    spec = formula["cancers"][formula.get("cancer_key_aliases", {}).get(key, key)]
    return r_tr.to_numpy(), r_te.to_numpy(), len(spec["features"])


def run_compact_refit(cancer, train, test, compact_feats):
    Xtr = train[compact_feats].values
    Xte = test[compact_feats].values
    scaler = StandardScaler().fit(Xtr)
    m = CoxPHSurvivalAnalysis(alpha=0.01)
    m.fit(scaler.transform(Xtr), make_surv(train))
    return m.predict(scaler.transform(Xtr)), m.predict(scaler.transform(Xte)), len(compact_feats)


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_delta(df_te, r_ref, r_other, B, rng):
    """Paired bootstrap of Delta C = C(ref) - C(other) on the test set."""
    t = df_te["OS_MONTHS"].to_numpy(float)
    e = df_te["Event_OS"].astype(bool).to_numpy()
    n = len(t)
    deltas, c_ref, c_oth = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if e[idx].sum() < 2:
            continue
        cr = concordance_index(t[idx], -r_ref[idx], e[idx])
        co = concordance_index(t[idx], -r_other[idx], e[idx])
        c_ref.append(cr); c_oth.append(co); deltas.append(cr - co)
    deltas = np.array(deltas)
    p = 2 * min((deltas < 0).mean(), (deltas > 0).mean())
    return {
        "delta_c_mean": float(deltas.mean()),
        "delta_c_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_c_ci_hi": float(np.percentile(deltas, 97.5)),
        "boot_p_two_sided": float(min(p, 1.0)),
        "c_ref_ci": [float(np.percentile(c_ref, 2.5)), float(np.percentile(c_ref, 97.5))],
        "c_other_ci": [float(np.percentile(c_oth, 2.5)), float(np.percentile(c_oth, 97.5))],
    }


# ----------------------------------------------------------------------------- driver
def run_cancer(cancer, formula, decision, B, with_clinical):
    print(f"\n=== {cancer} ===", flush=True)
    train, test, tr_ids, te_ids = load_split(cancer)
    key = CANCER_ALIAS.get(cancer, cancer)
    compact_feats = decision[decision["cancer"] == key]["feature_name_internal"].tolist()
    K = len(compact_feats)
    print(f"  train={len(train)} test={len(test)} K={K} feats={compact_feats}", flush=True)

    arms = {}
    r_tr, r_te, nf = run_compact_fixed(cancer, train, test, formula)
    arms["compact_fixed"] = (r_tr, r_te, nf)
    print(f"  compact_fixed done (nf={nf})", flush=True)

    r_tr, r_te, nf = run_compact_refit(cancer, train, test, compact_feats)
    arms["compact_refit"] = (r_tr, r_te, nf)
    print(f"  compact_refit done", flush=True)

    gs = retained_gene_set(cancer)
    r_tr, r_te, nf, ncand = run_gene_arm(cancer, train, test, tr_ids, te_ids, gs,
                                         with_clinical=with_clinical)
    arms["gene_restricted"] = (r_tr, r_te, nf)
    print(f"  gene_restricted done (nf={nf}/{ncand} cand)", flush=True)

    r_tr, r_te, nf, ncand = run_gene_arm(cancer, train, test, tr_ids, te_ids, None,
                                         with_clinical=with_clinical)
    arms["gene_unrestricted"] = (r_tr, r_te, nf)
    print(f"  gene_unrestricted done (nf={nf}/{ncand} cand)", flush=True)

    r_tr, r_te, nf, ncand = run_gene_arm(cancer, train, test, tr_ids, te_ids, gs,
                                         cap_k=K, with_clinical=with_clinical)
    arms["gene_capped_K"] = (r_tr, r_te, nf)
    print(f"  gene_capped_K done (nf={nf})", flush=True)

    y_tr = make_surv(train)
    rows = []
    for name, (rtr, rte, nf) in arms.items():
        m = compute_metrics(y_tr, train, test, rtr, rte)
        m.update({"cancer": cancer, "arm": name, "n_features": nf})
        rows.append(m)
        print(f"    {name:20s} nf={nf:3d}  C={m['c_index']:.4f}  "
              f"C-IPCW={m['c_ipcw']:.4f}  tdAUC24={m['td_auc_24']:.4f}  "
              f"Acc24={m['acc_24']:.4f}", flush=True)

    # paired bootstrap: every arm vs deployed compact_fixed
    rng = np.random.default_rng(SEED)
    ref_te = arms["compact_fixed"][1]
    boot = []
    for name, (_, rte, _) in arms.items():
        if name == "compact_fixed":
            continue
        d = bootstrap_delta(test, ref_te, rte, B, rng)
        d.update({"cancer": cancer, "comparison": f"compact_fixed_minus_{name}"})
        boot.append(d)
        print(f"    ΔC vs {name:18s} = {d['delta_c_mean']:+.4f} "
              f"[{d['delta_c_ci_lo']:+.4f},{d['delta_c_ci_hi']:+.4f}] "
              f"p={d['boot_p_two_sided']:.3f}", flush=True)
    return rows, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "figures_tables" / "gene_vs_pathway_headtohead"))
    ap.add_argument("--cancers", nargs="+", default=CANCERS)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--with-clinical", action="store_true",
                    help="add clinical fixed covariates to gene arms (robustness variant)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formula = yaml.safe_load(open(FORMULA))
    decision = pd.read_csv(DECISION)

    all_metrics, all_boot = [], []
    for c in args.cancers:
        rows, boot = run_cancer(c, formula, decision, args.boot, args.with_clinical)
        all_metrics += rows
        all_boot += boot

    mdf = pd.DataFrame(all_metrics)[
        ["cancer", "arm", "n_features", "c_index", "c_ipcw", "td_auc_24", "acc_24"]]
    bdf = pd.DataFrame(all_boot)
    tag = "_with_clinical" if args.with_clinical else ""
    mdf.to_csv(out / f"headtohead_metrics{tag}.csv", index=False)
    bdf.to_csv(out / f"headtohead_bootstrap_deltaC{tag}.csv", index=False)
    with open(out / f"headtohead{tag}.json", "w") as f:
        json.dump({"metrics": all_metrics, "bootstrap": all_boot}, f, indent=2)
    print(f"\nwrote {out}/headtohead_metrics{tag}.csv")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
