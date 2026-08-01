#!/usr/bin/env python3
"""
feature_to_gene_traceback.py
============================
Reverse-map each cancer's final 4-5 compact pathway features back to the
underlying genes ("which genes are actually inside these features?").

For every final compact feature (ibcga_run_records/final_feature_selection_decision.csv)
this decomposes:
  * structure       : pathway(s), variant type(s) used, aggregation logic
  * constituent genes : template gene set (TCGA 10-pathway) restricted to genes
                        actually observed altered (for the relevant variant type)
                        in that cancer's TRAIN cohort
  * per-gene weight   : train-cohort alteration frequency, univariate OS Cox
                        hazard ratio (HR) + p, and a contribution score
                        = frequency * |log HR|, ranked within the feature.

The contribution ranking answers the committee's question directly: e.g.
"PW_TP53_mut_rate_z is driven mainly by TP53 (and secondarily ATM/CHEK2)."

Alignment is done via the reproduced stratified split (shared with
gene_vs_pathway_headtohead.py) so gene events line up 1:1 with survival.

Run:
    python3 scripts/feature_to_gene_traceback.py \
        --out-dir figures_tables/feature_to_gene_traceback
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_3flag_enhanced import parse_10pathway, CANCER_PATHWAYS
import gene_vs_pathway_headtohead as HH   # reuse reproduce_sids / build_gene_events

DECISION = ROOT / "ibcga_run_records" / "final_feature_selection_decision.csv"
PATHWAY_XLSX = ROOT / "build_3flag_inputs" / "tcga_10_pathway.xlsx"
CANCERS = ["BRCA", "LUAD", "PAAD", "PRAD", "CRC"]
CANCER_ALIAS = {"BRCA": "IDC"}

# known pathway keys (longest first so multi-word keys match before splitting)
PW_KEYS = ["RTK_RAS", "Cell_Cycle", "TGF_Beta", "TP53", "PI3K", "Chromatin",
           "DDR", "MYC", "NOTCH", "NRF2", "HIPPO", "WNT"]
# suffix -> set of alteration-event types that feed the feature
SUFFIX_VARIANTS = {
    "mut_rate_z": ["mut"],
    "any_rate_z": ["mut", "amp", "del"],
    "amp_hit":    ["amp"],
    "del_hit":    ["del"],
    "sv_hit":     ["sv"],
    "zsum":       ["mut", "amp", "del", "sv"],
}
SUFFIX_LOGIC = {
    "mut_rate_z": "nonsilent-mutation rate over pathway genes (z-scored)",
    "any_rate_z": "fraction of pathway genes with mut/amp/del (z-scored)",
    "amp_hit":    "binary: any pathway gene amplified (CNA>=+2)",
    "del_hit":    "binary: any pathway gene deep-deleted (CNA<=-2)",
    "sv_hit":     "binary: any pathway gene with a structural variant",
    "zsum":       "composite: z(amp)+z(del)+z(sv)+mut_rate_z over pathway genes",
}


def parse_feature(name, cancer):
    """Return (pathways:list, variants:list, logic:str) for a PW_* feature name."""
    if name == "PW_HIT_COUNT":
        return (list(CANCER_PATHWAYS[cancer]), ["mut", "amp", "del", "sv"],
                "count of retained pathways with >=1 alteration (all pathways)")
    body = name[3:] if name.startswith("PW_") else name
    for suf in sorted(SUFFIX_VARIANTS, key=len, reverse=True):
        if body.endswith("_" + suf) or body == suf:
            pw = body[: -(len(suf) + 1)]
            return ([pw], SUFFIX_VARIANTS[suf], SUFFIX_LOGIC[suf])
    raise ValueError(f"cannot parse feature {name!r}")


def gene_indicator(events, gene, variants):
    """Binary Series: gene altered by any of `variants` (over samples in events)."""
    cols = [f"g_{gene}_{v}" for v in variants if f"g_{gene}_{v}" in events.columns]
    if not cols:
        return None
    return (events[cols].sum(axis=1) > 0).astype(int)


def univariate_hr(ind, dur, evt):
    """Univariate CoxPH HR + p for a binary indicator. Returns (hr, p, n_alt)."""
    n_alt = int(ind.sum())
    if n_alt < 5 or n_alt > len(ind) - 5:
        return (np.nan, np.nan, n_alt)
    d = pd.DataFrame({"x": ind.values, "T": dur.values, "E": evt.values})
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(d, duration_col="T", event_col="E")
        return (float(np.exp(cph.params_["x"])), float(cph.summary.loc["x", "p"]), n_alt)
    except Exception:
        return (np.nan, np.nan, n_alt)


def run_cancer(cancer, decision):
    key = CANCER_ALIAS.get(cancer, cancer)
    feats = decision[decision["cancer"] == key]
    train, _test, tr_sids, te_sids = HH.load_split(cancer)
    dur = train["OS_MONTHS"].astype(float).reset_index(drop=True)
    evt = train["Event_OS"].astype(bool).reset_index(drop=True)

    pw_genes = parse_10pathway(str(PATHWAY_XLSX))
    # build all gene events once (all panel genes) for train samples, in train row order
    ev_all = HH.build_gene_events(cancer, tr_sids + te_sids, None).loc[tr_sids].reset_index(drop=True)

    rows = []
    for _, fr in feats.iterrows():
        fname = fr["feature_name_internal"]
        disp = fr["feature_name_display"]
        beta = float(fr["beta"])
        pws, variants, logic = parse_feature(fname, cancer)
        gene_set = set()
        for p in pws:
            gene_set |= pw_genes[p]["all"]
        for g in sorted(gene_set):
            ind = gene_indicator(ev_all, g, variants)
            if ind is None or ind.sum() == 0:
                continue                       # not observed altered in this cohort
            freq = float(ind.mean())
            hr, p, n_alt = univariate_hr(ind, dur, evt)
            abs_loghr = abs(np.log(hr)) if (hr and hr > 0 and np.isfinite(hr)) else np.nan
            contrib = freq * abs_loghr if np.isfinite(abs_loghr) else np.nan
            rows.append({
                "cancer": cancer, "feature_internal": fname, "feature_display": disp,
                "beta": beta, "pathways": ";".join(pws),
                "variant_types": "/".join(variants), "logic": logic,
                "gene": g, "n_altered_train": n_alt, "freq_train": round(freq, 4),
                "univ_HR": round(hr, 3) if np.isfinite(hr) else np.nan,
                "univ_p": p,
                "abs_logHR": round(abs_loghr, 4) if np.isfinite(abs_loghr) else np.nan,
                "contribution": round(contrib, 4) if np.isfinite(contrib) else np.nan,
            })
    df = pd.DataFrame(rows)
    # rank genes within each feature by contribution (desc)
    df["rank_in_feature"] = (
        df.groupby("feature_internal")["contribution"]
          .rank(ascending=False, method="min"))
    df = df.sort_values(["feature_internal", "contribution"],
                        ascending=[True, False]).reset_index(drop=True)
    return df


HR_MIN_N = 10   # min altered-sample count for a stable univariate HR in the unweighted view


def classify(share):
    if not np.isfinite(share):
        return "n/a"
    if share >= 0.90:
        return "single-gene"
    if share >= 0.50:
        return "driver-anchored"
    return "multi-gene"


def summarize(df, top=6):
    out = []
    for (cancer, fint), sub in df.groupby(["cancer", "feature_internal"], sort=False):
        sub = sub.sort_values("contribution", ascending=False)
        c = sub.dropna(subset=["contribution"])
        disp = sub["feature_display"].iloc[0]
        # frequency-weighted contribution view
        top_str = "; ".join(
            f"{r.gene}(f={r.freq_train:.2f},HR={r.univ_HR:.2f})"
            for r in c.head(top).itertuples())
        # top-gene dominance
        tot = c["contribution"].sum()
        top1_gene = c["gene"].iloc[0] if len(c) else ""
        top1_share = (c["contribution"].iloc[0] / tot) if (len(c) and tot > 0) else np.nan
        # unweighted prognostic-strength view (stable genes only, |log HR| ranked)
        stable = sub[(sub["n_altered_train"] >= HR_MIN_N) & sub["abs_logHR"].notna()] \
            .sort_values("abs_logHR", ascending=False)
        byhr_str = "; ".join(
            f"{r.gene}(HR={r.univ_HR:.2f},f={r.freq_train:.2f},n={int(r.n_altered_train)})"
            for r in stable.head(top).itertuples())
        out.append({
            "cancer": cancer, "feature_internal": fint, "feature_display": disp,
            "beta": sub["beta"].iloc[0], "pathways": sub["pathways"].iloc[0],
            "variant_types": sub["variant_types"].iloc[0],
            "n_constituent_genes_observed": sub["gene"].nunique(),
            "top1_gene": top1_gene,
            "top1_share_pct": round(top1_share * 100, 1) if np.isfinite(top1_share) else np.nan,
            "dominance_class": classify(top1_share),
            "top_genes_by_contribution": top_str,
            "top_genes_by_abs_logHR_stable": byhr_str,
        })
    return pd.DataFrame(out)


def make_figure(df, summ, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from figure_style import apply_arial_style
        apply_arial_style()
    except Exception:
        pass

    share_lookup = {(r.cancer, r.feature_internal): (r.top1_share_pct, r.dominance_class)
                    for r in summ.itertuples()}

    # ---- Figure 1: frequency-weighted contribution (freq x |log HR|) ----
    for cancer, csub in df.groupby("cancer"):
        feats = list(dict.fromkeys(csub["feature_internal"]))
        n = len(feats)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.4), squeeze=False)
        axes = axes[0]
        for ax, f in zip(axes, feats):
            s = (csub[csub["feature_internal"] == f]
                 .dropna(subset=["contribution"])       # NaN HR genes sort last -> would hijack tail()
                 .sort_values("contribution", ascending=True).tail(8))
            disp = s["feature_display"].iloc[0]
            share, klass = share_lookup.get((cancer, f), (np.nan, ""))
            colors = ["#c0392b" if hr >= 1 else "#2874a6" for hr in s["univ_HR"].fillna(1)]
            ax.barh(s["gene"], s["contribution"], color=colors)
            sub_t = f"top gene {share:.0f}% · {klass}" if np.isfinite(share) else klass
            ax.set_title(f"{disp}\n{sub_t}", fontsize=8.5)
            ax.set_xlabel("contribution\n(freq x |log HR|)", fontsize=7)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{cancer}: compact features -> top constituent genes, frequency-weighted "
                     f"(red HR>=1 worse OS, blue better)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f"{cancer}_feature_to_gene.png", dpi=200)
        fig.savefig(out_dir / f"{cancer}_feature_to_gene.pdf")
        plt.close(fig)

    # ---- Figure 2: unweighted prognostic strength (|log HR|, stable genes n>=HR_MIN_N) ----
    # Surfaces moderately-rare-but-prognostic genes that the frequency weighting hides.
    for cancer, csub in df.groupby("cancer"):
        feats = list(dict.fromkeys(csub["feature_internal"]))
        n = len(feats)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.4), squeeze=False)
        axes = axes[0]
        for ax, f in zip(axes, feats):
            s = (csub[(csub["feature_internal"] == f)
                      & (csub["n_altered_train"] >= HR_MIN_N)]
                 .dropna(subset=["abs_logHR"])
                 .sort_values("abs_logHR", ascending=True).tail(8))
            disp = s["feature_display"].iloc[0] if len(s) else \
                csub[csub["feature_internal"] == f]["feature_display"].iloc[0]
            colors = ["#c0392b" if hr >= 1 else "#2874a6" for hr in s["univ_HR"].fillna(1)]
            ax.barh(s["gene"], s["abs_logHR"], color=colors)
            ax.set_title(disp, fontsize=8.5)
            ax.set_xlabel("|log HR|  (unweighted)", fontsize=7)
            ax.tick_params(labelsize=7)
            if len(s) == 0:
                ax.text(0.5, 0.5, f"no gene\nwith n>={HR_MIN_N}", ha="center",
                        va="center", fontsize=8, transform=ax.transAxes)
        fig.suptitle(f"{cancer}: constituent genes by unweighted prognostic strength "
                     f"|log HR| (n>={HR_MIN_N}; red worse OS, blue better)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f"{cancer}_feature_to_gene_byHR.png", dpi=200)
        fig.savefig(out_dir / f"{cancer}_feature_to_gene_byHR.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "figures_tables" / "feature_to_gene_traceback"))
    ap.add_argument("--cancers", nargs="+", default=CANCERS)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    decision = pd.read_csv(DECISION)

    parts = []
    for c in args.cancers:
        print(f"=== {c} ===", flush=True)
        d = run_cancer(c, decision)
        parts.append(d)
        for f, sub in d.groupby("feature_internal", sort=False):
            top = sub.sort_values("contribution", ascending=False).head(4)
            top_s = ", ".join(f"{r.gene}({r.freq_train:.2f},HR{r.univ_HR:.2f})"
                              for r in top.itertuples() if np.isfinite(r.univ_HR))
            print(f"  {sub['feature_display'].iloc[0]:28s} <- {top_s}", flush=True)

    df = pd.concat(parts, ignore_index=True)
    summ = summarize(df)
    df.to_csv(out / "feature_to_gene_long.csv", index=False)
    summ.to_csv(out / "feature_to_gene_summary.csv", index=False)
    make_figure(df, summ, out)
    print(f"\nwrote {out}/feature_to_gene_long.csv  ({len(df)} rows)")
    print(f"wrote {out}/feature_to_gene_summary.csv")


if __name__ == "__main__":
    main()
