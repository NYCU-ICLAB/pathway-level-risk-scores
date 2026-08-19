"""genie_bpc_sensitivity_cox.py  (Tasks B1 + B2 + B3)
=====================================================
Sensitivity analyses on the 5 GENIE BPC cohorts, all using the fixed-formula
compact pathway score (no external retraining):

  B1  BrCa subtype-adjusted Cox
        OS ~ compact_score + age + sex_male + stage + met_sample + inst
              + BCA_SUBTYPE dummies (TNBC, HR+/HER2+, HR-/HER2+; baseline HR+/HER2-)
        and an alternative with ER/PR/HER2_pos individually.

  B2  Stage subgroup forest plot (all 5 cohorts)
        Split each cohort into Stage I-III vs Stage IV (matching MSK CHORD's
        binary STAGE_HIGHEST_RECORDED encoding), fit Cox(OS ~ compact_score
        + age + sex + met_sample + inst) within each subgroup, report
        compact_score HR per (cohort × stage subgroup).

  B3  First-line treatment-adjusted Cox (all 5 cohorts)
        Parse the patient's earliest START_DATE >= 0 regimen, classify into
        cancer-specific drug-class buckets, and add the dummy(s) to the
        multivariable Cox. Report compact_score adjusted HR before/after
        treatment adjustment.

Outputs:
  figures_tables/genie_bpc_sensitivity/B1_brca_subtype_cox.csv
  figures_tables/genie_bpc_sensitivity/B2_stage_subgroup.csv
  figures_tables/genie_bpc_sensitivity/B2_stage_forest.{png,pdf}
  figures_tables/genie_bpc_sensitivity/B3_treatment_classification_audit.csv
  figures_tables/genie_bpc_sensitivity/B3_treatment_adjusted_cox.csv
  figures_tables/genie_bpc_sensitivity/B3_treatment_forest.{png,pdf}
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from figure_style import apply_arial_style
apply_arial_style()
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW_BASE = ROOT / "external_validation" / "raw_data" / "GENIE_BPC"
OUT_BASE = ROOT / "external_validation" / "outputs"
OUT_DIR = ROOT / "figures_tables" / "genie_bpc_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORMULA = ROOT / "external_validation" / "zscore_params" / "compact_score_formula.yaml"

COHORTS = {  # cohort_key -> (cancer, subdir, features_csv_name)
    "BrCa":     ("BRCA", "BrCa",     "GENIE_BPC_BrCa_pathway_features.csv"),
    "NSCLC":    ("LUAD", "NSCLC",    "GENIE_BPC_NSCLC_pathway_features.csv"),
    "PANC":     ("PAAD", "PANC",     "GENIE_BPC_PANC_pathway_features.csv"),
    "Prostate": ("PRAD", "Prostate", "GENIE_BPC_Prostate_pathway_features.csv"),
    "CRC":      ("CRC",  "CRC",      "GENIE_BPC_CRC_pathway_features.csv"),
}
DISPLAY_CANCER = {"BRCA": "IDC"}

# Cancer-specific first-line treatment classification — keyword regex on AGENT/REGIMEN
TX_CLASS = {
    "BRCA": [
        ("Hormone", r"tamoxifen|anastrozole|letrozole|exemestane|fulvestrant|"
                     r"palbociclib|ribociclib|abemaciclib|goserelin|leuprolide"),
        ("Targeted", r"trastuzumab|pertuzumab|lapatinib|tucatinib|t-dm1|t-dxd|"
                      r"trastuzumab emtansine|trastuzumab deruxtecan|neratinib"),
        ("Immuno",   r"atezolizumab|pembrolizumab|nivolumab|durvalumab"),
        ("Chemo",    r"cyclophosphamide|paclitaxel|docetaxel|doxorubicin|"
                      r"epirubicin|carboplatin|cisplatin|capecitabine|"
                      r"gemcitabine|eribulin|vinorelbine|fluorouracil|methotrexate"),
    ],
    "LUAD": [
        ("Targeted", r"erlotinib|gefitinib|afatinib|osimertinib|dacomitinib|"
                      r"crizotinib|alectinib|ceritinib|brigatinib|lorlatinib|"
                      r"entrectinib|capmatinib|tepotinib|selpercatinib|"
                      r"pralsetinib|dabrafenib|trametinib|sotorasib|adagrasib"),
        ("Immuno",   r"nivolumab|pembrolizumab|atezolizumab|durvalumab|ipilimumab|cemiplimab"),
        ("Chemo",    r"carboplatin|cisplatin|pemetrexed|paclitaxel|docetaxel|"
                      r"gemcitabine|etoposide|vinorelbine|bevacizumab"),
    ],
    "PAAD": [
        ("FOLFIRINOX", r"folfirinox|(?=.*fluorouracil)(?=.*irinotecan)(?=.*oxaliplatin)"),
        ("GemNab",     r"(?=.*gemcitabine).*nab.?paclitaxel|(?=.*nabpaclitaxel).*gemcitabine"),
        ("Chemo",      r"gemcitabine|fluorouracil|capecitabine|paclitaxel|"
                        r"oxaliplatin|irinotecan|leucovorin|nabpaclitaxel"),
        ("Targeted",   r"erlotinib|olaparib|larotrectinib|entrectinib"),
        ("Immuno",     r"nivolumab|pembrolizumab|atezolizumab|durvalumab"),
    ],
    "PRAD": [
        ("ADT",       r"leuprolide|goserelin|degarelix|triptorelin|bicalutamide|"
                       r"flutamide|nilutamide|cyproterone"),
        ("ARSi",      r"abiraterone|enzalutamide|apalutamide|darolutamide"),
        ("Chemo",     r"docetaxel|cabazitaxel|mitoxantrone|estramustine"),
        ("Targeted",  r"olaparib|rucaparib|niraparib|talazoparib|sipuleucel|"
                       r"radium|lutetium"),
        ("Immuno",    r"pembrolizumab|nivolumab|ipilimumab"),
    ],
    "CRC": [
        ("Biologic",  r"bevacizumab|cetuximab|panitumumab|ramucirumab|aflibercept"),
        ("Immuno",    r"nivolumab|pembrolizumab|ipilimumab|atezolizumab|durvalumab"),
        ("Targeted",  r"regorafenib|trifluridine|encorafenib|binimetinib|"
                       r"trastuzumab|larotrectinib|entrectinib"),
        ("Chemo",     r"fluorouracil|oxaliplatin|irinotecan|capecitabine|leucovorin"),
    ],
}


def parse_stage(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper().replace("STAGE", "").strip()
    m = re.match(r"^(IV|III|II|I)", s)
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(m.group(1)) if m else np.nan


def sex_to_male(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    return 1.0 if s.startswith("m") else (0.0 if s.startswith("f") else np.nan)


def classify_first_line(regimen, agent, cancer):
    """Return one of the TX_CLASS labels for given cancer, or 'Other' / 'No treatment'."""
    if not regimen and not agent:
        return "No treatment"
    text = f"{regimen or ''} {agent or ''}".lower()
    rules = TX_CLASS.get(cancer, [])
    for label, pat in rules:
        if re.search(pat, text):
            return label
    return "Other"


def first_line_tx_per_patient(cohort_subdir, cancer):
    tx = pd.read_csv(RAW_BASE / cohort_subdir / "data_timeline_treatment.txt",
                     sep="\t", low_memory=False)
    tx = tx[tx["START_DATE"] >= 0].copy()
    tx["regimen_str"] = tx["REGIMEN"].astype(str)
    tx["agent_str"] = tx["AGENT"].astype(str)
    tx = tx.sort_values(["PATIENT_ID", "START_DATE"])
    first = tx.groupby("PATIENT_ID").first().reset_index()
    first["tx_first_line"] = first.apply(
        lambda r: classify_first_line(r["regimen_str"], r["agent_str"], cancer), axis=1
    )
    return first[["PATIENT_ID", "tx_first_line", "regimen_str", "agent_str"]]


def load_cohort(cohort_key, formula):
    cancer, subdir, fcsv = COHORTS[cohort_key]
    feat = pd.read_csv(OUT_BASE / f"GENIE_BPC_{cohort_key}" / fcsv)
    spec = formula["cancers"][cancer]
    feats = [f["name"] for f in spec["features"]]
    betas = np.array([float(f["beta"]) for f in spec["features"]])
    feat["compact_score"] = (feat[feats].values * betas).sum(axis=1)
    # report HR per 1-SD of the compact score (per-cohort SD), consistent with the main-text
    # per-1-SD hazard ratios rather than the raw per-unit linear predictor
    feat["compact_score"] = feat["compact_score"] / feat["compact_score"].std(ddof=1)

    pat = pd.read_csv(RAW_BASE / subdir / "clinical_patient_nonMSK.csv", low_memory=False)
    df = feat.copy()
    df["age"] = pd.to_numeric(df["AGE_AT_SEQUENCING"], errors="coerce")
    df["met_sample"] = df["SAMPLE_TYPE_DETAILED"].astype(str).str.contains(
        "metastasis", case=False, na=False).astype(int)
    df["inst_UHN"] = (df["INSTITUTION"] == "UHN").astype(int)
    df["inst_VICC"] = (df["INSTITUTION"] == "VICC").astype(int)
    df = df.merge(pat, on="PATIENT_ID", how="left", suffixes=("", "_pat"))
    df["sex_male"] = df["SEX"].apply(sex_to_male)
    df["stage_ord"] = df["STAGE_DX"].apply(parse_stage)
    df["stage4"] = (df["stage_ord"] >= 4).astype("Int64")   # IV
    df["stage_low3"] = (df["stage_ord"] <= 3).astype("Int64")  # I-III
    tx = first_line_tx_per_patient(subdir, cancer)
    df = df.merge(tx, on="PATIENT_ID", how="left")
    df["tx_first_line"] = df["tx_first_line"].fillna("No treatment")
    return df, cancer


def fit_cox(df, covs):
    use = df[["OS_MONTHS", "Event_OS"] + covs].replace([np.inf, -np.inf], np.nan).dropna()
    keep = [c for c in covs if use[c].nunique() > 1]
    use = use[["OS_MONTHS", "Event_OS"] + keep]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(use, duration_col="OS_MONTHS", event_col="Event_OS", show_progress=False)
    return cph, use, keep


def score_row(cph, n, ev, extras=None):
    if "compact_score" not in cph.summary.index:
        return None
    s = cph.summary.loc["compact_score"]
    row = {
        "n": int(n), "events": int(ev),
        "score_HR": float(s["exp(coef)"]),
        "score_lower95": float(s["exp(coef) lower 95%"]),
        "score_upper95": float(s["exp(coef) upper 95%"]),
        "score_p": float(s["p"]),
        "cindex": float(cph.concordance_index_),
    }
    if extras:
        row.update(extras)
    return row


# ── B1 ─────────────────────────────────────────────────────────────────────────
def yes_no_pos(v):
    if pd.isna(v):
        return np.nan
    return 1.0 if "positive" in str(v).lower() else (0.0 if "negative" in str(v).lower() else np.nan)


def task_B1(df_brca):
    base = ["compact_score", "age", "sex_male", "stage_ord", "met_sample",
            "inst_VICC"]
    out = []
    # baseline (no subtype)
    cph, use, keep = fit_cox(df_brca, base)
    r = score_row(cph, len(use), use["Event_OS"].sum(),
                  extras={"model": "Baseline (no subtype)", "covariates": ", ".join(keep)})
    out.append(r)

    # BCA_SUBTYPE 4-level dummies (HR+/HER2- = baseline)
    sub = df_brca.copy()
    sub["BCA_SUBTYPE"] = sub["BCA_SUBTYPE"].astype(str)
    for lvl in ["Triple Negative", "HR+, HER2+", "HR-, HER2+"]:
        sub[f"subtype_{lvl}"] = (sub["BCA_SUBTYPE"] == lvl).astype(int)
    sub_cols = [f"subtype_{l}" for l in ["Triple Negative", "HR+, HER2+", "HR-, HER2+"]]
    cph, use, keep = fit_cox(sub, base + sub_cols)
    r = score_row(cph, len(use), use["Event_OS"].sum(),
                  extras={"model": "+ BCA_SUBTYPE dummies", "covariates": ", ".join(keep)})
    out.append(r)

    # ER/PR/HER2 binaries
    sub["ER_pos"]   = sub["CA_BCA_ER"].apply(yes_no_pos)
    sub["PR_pos"]   = sub["CA_BCA_PR"].apply(yes_no_pos)
    sub["HER2_pos"] = sub["CA_BCA_HER_SUMM"].apply(yes_no_pos)
    cph, use, keep = fit_cox(sub, base + ["ER_pos", "PR_pos", "HER2_pos"])
    r = score_row(cph, len(use), use["Event_OS"].sum(),
                  extras={"model": "+ ER/PR/HER2 binary", "covariates": ", ".join(keep)})
    out.append(r)

    return pd.DataFrame(out)


# ── B2 ─────────────────────────────────────────────────────────────────────────
def task_B2(df, cancer, cohort_key):
    covs = ["compact_score", "age", "sex_male", "met_sample", "inst_VICC"]
    out = []
    # Overall
    cph, use, _ = fit_cox(df.assign(stage_grp="Overall"), covs)
    out.append(score_row(cph, len(use), use["Event_OS"].sum(),
                          extras={"cohort": cohort_key, "cancer": cancer, "stage_grp": "Overall"}))
    # Stage I-III
    low = df[df["stage_low3"] == 1]
    cph, use, _ = fit_cox(low, covs)
    if cph is not None:
        out.append(score_row(cph, len(use), use["Event_OS"].sum(),
                              extras={"cohort": cohort_key, "cancer": cancer, "stage_grp": "Stage I-III"}))
    # Stage IV
    high = df[df["stage4"] == 1]
    if len(high) >= 30 and high["Event_OS"].sum() >= 10:
        cph, use, _ = fit_cox(high, covs)
        if cph is not None:
            out.append(score_row(cph, len(use), use["Event_OS"].sum(),
                                  extras={"cohort": cohort_key, "cancer": cancer, "stage_grp": "Stage IV"}))
    return [r for r in out if r is not None]


# ── B3 ─────────────────────────────────────────────────────────────────────────
def task_B3(df, cancer, cohort_key):
    base = ["compact_score", "age", "sex_male", "stage_ord", "met_sample", "inst_VICC"]
    out = []
    # Unadjusted (no tx)
    cph, use, _ = fit_cox(df, base)
    out.append(score_row(cph, len(use), use["Event_OS"].sum(),
                          extras={"cohort": cohort_key, "cancer": cancer,
                                  "model": "Without tx"}))
    # With tx dummies
    cats = [c for c in df["tx_first_line"].value_counts().index if c != "No treatment"]
    sub = df.copy()
    tx_cols = []
    for c in cats:
        col = f"tx_{c.replace(' ', '_')}"
        sub[col] = (sub["tx_first_line"] == c).astype(int)
        tx_cols.append(col)
    cph, use, _ = fit_cox(sub, base + tx_cols)
    out.append(score_row(cph, len(use), use["Event_OS"].sum(),
                          extras={"cohort": cohort_key, "cancer": cancer,
                                  "model": "With tx adjustment",
                                  "tx_categories": ", ".join(cats)}))
    return out, sub[["PATIENT_ID", "tx_first_line"]]


def plot_forest(rows, title, xlabel, path, label_col, group_col=None):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    label_values = df[label_col].astype(str).map(lambda x: DISPLAY_CANCER.get(x, x))
    if group_col:
        df["label"] = label_values + " | " + df[group_col].astype(str)
    else:
        df["label"] = label_values
    df = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, max(2.8, 0.5 * len(df) + 1.6)), constrained_layout=True)
    y = np.arange(len(df))
    ax.errorbar(df["score_HR"], y,
                xerr=[df["score_HR"] - df["score_lower95"], df["score_upper95"] - df["score_HR"]],
                fmt="o", color="#234", ecolor="#789", capsize=3, markersize=6)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}\nn={r.n}, ev={r.events}" for r in df.itertuples()])
    ax.set_xlabel(xlabel)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    xmax = ax.get_xlim()[1]
    for i, r in enumerate(df.itertuples()):
        ax.text(xmax, i,
                f"  HR={r.score_HR:.2f} ({r.score_lower95:.2f}-{r.score_upper95:.2f}), p={r.score_p:.2g}",
                va="center", fontsize=8)
    fig.savefig(path, dpi=220)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)


def main():
    with FORMULA.open() as f:
        formula = yaml.safe_load(f)

    b2_rows, b3_rows = [], []
    tx_audit = []

    for ck in COHORTS:
        print(f"\n========== GENIE BPC {ck} ==========")
        df, cancer = load_cohort(ck, formula)
        print(f"  n={len(df)}, events={int(df['Event_OS'].sum())}")
        print(f"  Stage dist: I-III={(df['stage_low3']==1).sum()}, IV={(df['stage4']==1).sum()}, "
              f"NaN={df['stage_ord'].isna().sum()}")
        print(f"  First-line tx: {df['tx_first_line'].value_counts().to_dict()}")
        tx_audit.append(df[["PATIENT_ID", "tx_first_line", "regimen_str", "agent_str"]].assign(cohort=ck))

        if ck == "BrCa":
            b1 = task_B1(df)
            b1["cohort"] = ck
            print("\n[B1 BRCA subtype-adjusted]")
            print(b1[["model", "n", "events", "score_HR", "score_lower95",
                     "score_upper95", "score_p", "cindex"]].to_string(index=False))
            b1.to_csv(OUT_DIR / "B1_brca_subtype_cox.csv", index=False)

        b2_rows.extend(task_B2(df, cancer, ck))
        b3_pair, _ = task_B3(df, cancer, ck)
        b3_rows.extend(b3_pair)

    pd.DataFrame(b2_rows).to_csv(OUT_DIR / "B2_stage_subgroup.csv", index=False)
    pd.DataFrame(b3_rows).to_csv(OUT_DIR / "B3_treatment_adjusted_cox.csv", index=False)
    pd.concat(tx_audit, ignore_index=True).to_csv(
        OUT_DIR / "B3_treatment_classification_audit.csv", index=False)

    b2_stage_rows = [r for r in b2_rows if r and r.get("stage_grp") != "Overall"]
    plot_forest(
        b2_stage_rows,
        title=r"$\bf{c}$  External GENIE BPC: compact score HR by stage subgroup",
        xlabel="HR per 1-SD increase in compact score",
        path=OUT_DIR / "B2_stage_forest.png",
        label_col="cancer", group_col="stage_grp",
    )
    # Build B3 forest data: show only "With tx adjustment" rows
    b3_with_tx = [r for r in b3_rows if r and r.get("model") == "With tx adjustment"]
    plot_forest(
        b3_with_tx,
        title="GENIE BPC: compact_score HR after treatment adjustment (5 cohorts)\n"
              "(adjusted for age, sex, stage, met_sample, inst + tx dummies)",
        xlabel="HR per 1-SD increase in compact score",
        path=OUT_DIR / "B3_treatment_forest.png",
        label_col="cancer",
    )
    print(f"\nwrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
