#!/usr/bin/env python3
"""
leave_one_gene_out.py
=====================
Leave-one-gene-out (LOGO) decomposition of each compact pathway feature.

Answers, quantitatively, the committee/reviewer question:
    "Is this pathway feature just a proxy for one dominant gene, or does it
     genuinely aggregate multiple individually-sparse gene alterations?"

Method (self-consistent, no refitting of beta):
  * Rebuild every compact feature from SAMPLE_ID-aligned gene events
    (mut/amp/del/sv) using the published construction rules. Fidelity vs the
    original model_input feature is high (binary hits 1.00; rate corr ~0.99).
  * Baseline compact score = sum_k beta_k * feature_k(full gene set), with the
    published beta. Report its test C-index next to the published score's for
    fidelity.
  * LOGO: for each feature f and each constituent gene g, rebuild ONLY f with g
    removed from its gene set, recompute the compact score, and measure
        dC_logo(f,g) = C(full) - C(drop g)      [test set]
    plus reclassification% = fraction of test patients whose high/low risk
    group (train-median cutoff) flips.
  * Whole-feature removal dC gives the feature's total contribution, so a
    single gene's dC can be read against it: if one gene's dC ~ the whole
    feature's dC -> single-gene-driven; if every gene's dC is small but the
    whole feature's dC is large -> distributed multi-gene signal.

Run:
    python3 scripts/leave_one_gene_out.py --out-dir figures_tables/leave_one_gene_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import gene_vs_pathway_headtohead as HH
from build_3flag_enhanced import parse_10pathway, CANCER_PATHWAYS
import feature_to_gene_traceback as FT

CANCERS = ["BRCA", "LUAD", "PAAD", "PRAD", "CRC"]
CANCER_ALIAS = {"BRCA": "IDC"}
DECISION = ROOT / "ibcga_run_records" / "final_feature_selection_decision.csv"
MIN_N = 5  # skip genes altered in <5 train samples (ΔC noise), still counted in feature build


class FeatureBuilder:
    """Rebuild compact features from SAMPLE_ID-aligned gene events."""

    def __init__(self, cancer):
        self.cancer = cancer
        self.train, self.test, self.tr, self.te = HH.load_split(cancer)
        self.ev = HH.build_gene_events(cancer, self.tr + self.te, None).astype(float)
        self.pw = parse_10pathway(str(HH.PATHWAY_XLSX))
        self.idx = self.ev.index

    def _g(self, gene, typ):
        c = f"g_{gene}_{typ}"
        return (self.ev[c] > 0).to_numpy() if c in self.ev.columns else np.zeros(len(self.ev), bool)

    def any_event(self, genes, types):
        m = np.zeros(len(self.ev), bool)
        for g in genes:
            for t in types:
                m = m | self._g(g, t)
        return m.astype(float)

    def rate(self, genes, types):
        if not genes:
            return np.zeros(len(self.ev))
        cnt = np.zeros(len(self.ev))
        for g in genes:
            gi = np.zeros(len(self.ev), bool)
            for t in types:
                gi = gi | self._g(g, t)
            cnt = cnt + gi
        return cnt / len(genes)

    def _ztr(self, x):
        xs = pd.Series(x, index=self.idx)
        mu = xs.loc[self.tr].mean(); sd = xs.loc[self.tr].std(ddof=0)
        return ((xs - mu) / sd).to_numpy() if sd > 0 else np.zeros(len(x))

    def build(self, fname, drop_gene=None):
        """Return feature vector (all samples) for compact feature `fname`,
        optionally with `drop_gene` removed from its gene set."""
        pws, variants, _ = FT.parse_feature(fname, self.cancer)

        def gs(pw_key):
            s = set(self.pw[pw_key]["all"])
            if drop_gene:
                s.discard(drop_gene)
            return s

        if fname == "PW_HIT_COUNT":
            cnt = np.zeros(len(self.ev))
            for p in CANCER_PATHWAYS[self.cancer]:
                cnt = cnt + (self.any_event(gs(p), ["mut", "amp", "del", "sv"]) > 0)
            return cnt
        pk = pws[0]
        genes = gs(pk)
        if fname.endswith("_mut_rate_z"):
            return self._ztr(self.rate(genes, ["mut"]))
        if fname.endswith("_any_rate_z"):
            return self._ztr(self.rate(genes, ["mut", "amp", "del"]))
        if fname.endswith("_amp_hit"):
            return self.any_event(genes, ["amp"])
        if fname.endswith("_del_hit"):
            return self.any_event(genes, ["del"])
        if fname.endswith("_sv_hit"):
            return self.any_event(genes, ["sv"])
        if fname.endswith("_zsum"):
            return (self._ztr(self.any_event(genes, ["amp"]))
                    + self._ztr(self.any_event(genes, ["del"]))
                    + self._ztr(self.any_event(genes, ["sv"]))
                    + self._ztr(self.rate(genes, ["mut"])))
        raise ValueError(fname)


def test_cindex(builder, score_all):
    s = pd.Series(score_all, index=builder.idx).loc[builder.te].to_numpy()
    t = builder.test["OS_MONTHS"].to_numpy(float)
    e = builder.test["Event_OS"].astype(bool).to_numpy()
    return float(concordance_index(t, -s, e))


def reclass_pct(builder, score_full, score_drop):
    sf = pd.Series(score_full, index=builder.idx)
    sd = pd.Series(score_drop, index=builder.idx)
    cut = sf.loc[builder.tr].median()
    gf = (sf.loc[builder.te] > cut).astype(int)
    gd = (sd.loc[builder.te] > cut).astype(int)
    return float((gf != gd).mean())


def run_cancer(cancer, decision):
    key = CANCER_ALIAS.get(cancer, cancer)
    feats = decision[decision["cancer"] == key]
    fb = FeatureBuilder(cancer)
    betas = {r.feature_name_internal: float(r.beta) for r in feats.itertuples()}
    disp = {r.feature_name_internal: r.feature_name_display for r in feats.itertuples()}

    base_feat = {f: fb.build(f) for f in betas}
    score_full = sum(betas[f] * base_feat[f] for f in betas)
    c_full = test_cindex(fb, score_full)

    # fidelity vs published compact score
    from compact_score.apply_compact_score import apply_score, load_formula
    _, pub_risk = apply_score(fb.test, key, load_formula())
    c_pub = float(concordance_index(fb.test["OS_MONTHS"], -pub_risk.to_numpy(),
                                    fb.test["Event_OS"].astype(bool)))
    print(f"\n=== {cancer} ===  rebuilt-score test C={c_full:.4f}  published C={c_pub:.4f}", flush=True)

    rows = []
    for f in betas:
        # whole-feature removal (set this feature to 0)
        score_wo = sum(betas[k] * base_feat[k] for k in betas if k != f)
        c_wo = test_cindex(fb, score_wo)
        dC_feature = c_full - c_wo
        # constituent genes
        pws, variants, _ = FT.parse_feature(f, cancer)
        gene_set = set()
        for p in pws:
            gene_set |= fb.pw[p]["all"]
        for g in sorted(gene_set):
            # prevalence over the feature's variant set
            alt = np.zeros(len(fb.ev), bool)
            for t in variants:
                alt = alt | fb._g(g, t)
            n_alt = int(pd.Series(alt, index=fb.idx).loc[fb.tr].sum())
            if n_alt < MIN_N:
                continue
            fdrop = fb.build(f, drop_gene=g)
            score_drop = score_full - betas[f] * base_feat[f] + betas[f] * fdrop
            c_drop = test_cindex(fb, score_drop)
            rows.append({
                "cancer": cancer, "feature_internal": f, "feature_display": disp[f],
                "gene": g, "n_altered_train": n_alt,
                "freq_train": round(n_alt / len(fb.tr), 4),
                "dC_logo": round(c_full - c_drop, 5),
                "reclass_pct": round(reclass_pct(fb, score_full, score_drop), 4),
                "dC_whole_feature": round(dC_feature, 5),
            })
        print(f"  {disp[f]:28s} whole-feature ΔC={dC_feature:+.4f}", flush=True)
    return pd.DataFrame(rows), c_full, c_pub


IMPACT_MIN = 0.005  # min whole-feature test ΔC for the single/multi ratio to be meaningful


def classify_feature(sub):
    """Classify how a feature's discrimination is distributed across its genes.
    Features whose whole-feature ΔC is tiny (<IMPACT_MIN) are 'low-impact': the
    single/multi ratio would divide by ~0 and is not meaningful."""
    whole = sub["dC_whole_feature"].iloc[0]
    top = sub["dC_logo"].max()
    if whole < IMPACT_MIN:
        return "low-impact", np.nan
    ratio = top / whole
    if ratio >= 0.70:
        return "single-gene-driven", ratio
    if ratio >= 0.40:
        return "few-gene", ratio
    return "distributed-multi-gene", ratio


def make_figure(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from figure_style import apply_arial_style
        apply_arial_style()
    except Exception:
        pass
    for cancer, csub in df.groupby("cancer"):
        feats = list(dict.fromkeys(csub["feature_internal"]))
        n = len(feats)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.4), squeeze=False)
        axes = axes[0]
        for ax, f in zip(axes, feats):
            s = csub[csub["feature_internal"] == f].sort_values("dC_logo").tail(8)
            disp = s["feature_display"].iloc[0] if len(s) else f
            whole = s["dC_whole_feature"].iloc[0] if len(s) else 0
            ax.barh(s["gene"], s["dC_logo"], color="#7d3c98")
            ax.axvline(whole, ls="--", color="#c0392b", lw=1,
                       label=f"whole-feature ΔC={whole:.3f}")
            ax.set_title(disp, fontsize=8.5)
            ax.set_xlabel("leave-one-gene-out ΔC-index", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="lower right")
        fig.suptitle(f"{cancer}: leave-one-gene-out — how much each gene's removal drops "
                     f"the compact score (red dashed = whole feature)", fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f"{cancer}_leave_one_gene_out.png", dpi=200)
        fig.savefig(out_dir / f"{cancer}_leave_one_gene_out.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "figures_tables" / "leave_one_gene_out"))
    ap.add_argument("--cancers", nargs="+", default=CANCERS)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    decision = pd.read_csv(DECISION)

    parts, fid = [], []
    for c in args.cancers:
        d, c_full, c_pub = run_cancer(c, decision)
        parts.append(d)
        fid.append({"cancer": c, "rebuilt_score_test_C": round(c_full, 4),
                    "published_score_test_C": round(c_pub, 4)})
    df = pd.concat(parts, ignore_index=True)
    # per-feature classification summary
    srows = []
    for (cancer, f), sub in df.groupby(["cancer", "feature_internal"], sort=False):
        klass, ratio = classify_feature(sub)
        top = sub.sort_values("dC_logo", ascending=False).iloc[0]
        srows.append({
            "cancer": cancer, "feature_internal": f,
            "feature_display": sub["feature_display"].iloc[0],
            "dC_whole_feature": sub["dC_whole_feature"].iloc[0],
            "top_gene": top["gene"], "top_gene_dC_logo": top["dC_logo"],
            "top_gene_share_of_feature": round(ratio, 3) if np.isfinite(ratio) else np.nan,
            "logo_class": klass,
        })
    summ = pd.DataFrame(srows)
    df.to_csv(out / "leave_one_gene_out_long.csv", index=False)
    summ.to_csv(out / "leave_one_gene_out_summary.csv", index=False)
    pd.DataFrame(fid).to_csv(out / "rebuild_fidelity.csv", index=False)
    make_figure(df, out)
    print("\n" + summ.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
