#!/usr/bin/env python3
"""
benchmark_common_coverage.py  (Batch D)
=======================================
Common-coverage external gene-level benchmark. Reviewer concern: on GENIE, gene
features whose gene is not on a cohort's panel were previously set to 0 ("not
assayed != wild-type"), which could bias the external gene-level comparison.

Fix (chosen option): restrict the gene-level models to gene-variant features that
are COMMON-COVERAGE — i.e. whose gene is actually assayed in the target GENIE
cohort (mutation genes = genes present in data_mutations; CNA genes = rows of
data_cna). The same common-coverage model is fit on MSK train and evaluated both
internally (MSK held-out) and externally (GENIE), so no unassayed->0 padding is
needed. The compact pathway score arm is unchanged.

Outputs (overwrite; originals backed up to *.prepanel.csv):
  figures_tables/gene_vs_pathway_external/benchmark_bootstrap_table.csv   (Table 3.6)
  figures_tables/gene_vs_pathway_external/combined_internal_external.csv   (Figure 3.6)
  figures_tables/gene_vs_pathway_external/panel_coverage_harmonization.csv (Table S11)
"""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import gene_vs_pathway_external as EX
import gene_vs_pathway_headtohead as HH

DISP = {"BRCA": "IDC", "LUAD": "LUAD", "PAAD": "PAAD", "PRAD": "PRAD", "CRC": "CRC"}
DECISION = ROOT / "ibcga_run_records" / "final_feature_selection_decision.csv"
OUT = ROOT / "figures_tables" / "gene_vs_pathway_external"
B = 1000
ARMS = ["compact_fixed", "gene_capped_K", "gene_restricted"]
ALAB = {"compact_fixed": "Pathway compact", "gene_capped_K": "Gene (matched-size ≤Kc)",
        "gene_restricted": "Gene (pathway genes)"}


def genie_coverage(raw_dir):
    """Genes assayed in a GENIE cohort: mutation panel proxy = genes in data_mutations;
    CNA panel = genes (rows) in data_cna."""
    raw = EX.RAW_BASE / raw_dir
    mut = pd.read_csv(raw / "data_mutations.tsv", sep="\t", low_memory=False)
    mut_genes = set(mut["Hugo_Symbol"].dropna().unique())
    cna = pd.read_csv(raw / "data_cna.tsv", sep="\t", low_memory=False)
    cna_genes = set(cna[cna.columns[0]].dropna().unique())
    return mut_genes, cna_genes


def evaluable(col, mut_genes, cna_genes):
    m = re.match(r"g_(.+)_(mut|amp|del|sv)$", col)
    if not m:
        return False
    g, t = m.group(1), m.group(2)
    if t == "mut":
        return g in mut_genes
    if t in ("amp", "del"):
        return g in cna_genes
    return False   # SV omitted externally (panels lack uniform callable SV)


def boot_ci(t, e, risk, rng):
    n = len(t); vals = []
    for _ in range(B):
        bi = rng.integers(0, n, n)
        if e[bi].sum() < 2:
            continue
        vals.append(concordance_index(t[bi], -risk[bi], e[bi]))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dec = pd.read_csv(DECISION); rng = np.random.default_rng(42)
    rows, comb, cov_rows, paired_rows = [], [], [], []
    for cancer, (raw_dir, out_dir, _onc) in EX.COHORTS.items():
        key = HH.CANCER_ALIAS.get(cancer, cancer)
        K = int((dec["cancer"] == key).sum())
        mut_genes, cna_genes = genie_coverage(raw_dir)
        train, test, tr, te = HH.load_split(cancer)
        ti = test["OS_MONTHS"].to_numpy(float); ei = test["Event_OS"].astype(bool).to_numpy()
        pw = pd.read_csv(EX.OUT_BASE / out_dir / f"{out_dir}_pathway_features.csv")
        pw = pw[(pw["OS_MONTHS"] > 0) & pw["Event_OS"].notna()].copy()
        sids = pw["SAMPLE_ID"].tolist()
        te_e = pw["OS_MONTHS"].to_numpy(float); ee = pw["Event_OS"].astype(bool).to_numpy()
        genie_ev = EX.build_genie_gene_events(raw_dir, sids)
        gs = HH.retained_gene_set(cancer)

        # candidate features (>=1% MSK train prevalence) and their common-coverage subset
        ev = HH.build_gene_events(cancer, tr + te, gs)
        prev = ev.loc[tr].mean(0)
        cand = prev[prev >= HH.PREVALENCE].index.tolist()
        cov = [c for c in cand if evaluable(c, mut_genes, cna_genes)]
        cov_rows.append({"Cancer": DISP[cancer], "Retained-pathway genes": len(gs),
                         "Candidate gene-variant features (MSK, >=1%)": len(cand),
                         "Common-coverage in GENIE": len(cov),
                         "Dropped (unassayed externally)": len(cand) - len(cov),
                         "Coverage %": round(100 * len(cov) / max(1, len(cand)), 1)})

        def frozen(cap):
            """Fit gene Coxnet on MSK train restricted to common-coverage features."""
            Xtr = ev.loc[tr, cov]; sc = StandardScaler().fit(Xtr.values)
            # matched-size arm uses the fairer CV-best-under-cap selection (not just densest)
            m, nu = HH.coxnet_select(sc.transform(Xtr.values), HH.make_surv(train), cap_k=cap,
                                     cap_mode="cv_best" if cap is not None else "densest")
            ri = m.predict(sc.transform(ev.loc[te, cov].values))
            re = EX.apply_frozen({"model": m, "scaler": sc, "cols": cov}, genie_ev)
            return ri, re, nu

        from compact_score.apply_compact_score import apply_score, load_formula
        _, cri = apply_score(test, key, load_formula())
        risks = {"compact_fixed": (cri.to_numpy(), pw["risk_score"].to_numpy(float), K)}
        risks["gene_capped_K"] = frozen(K)
        risks["gene_restricted"] = frozen(None)

        for arm in ARMS:
            ri, re, nf = risks[arm]
            im, ilo, ihi = boot_ci(ti, ei, np.asarray(ri), rng)
            em, elo, ehi = boot_ci(te_e, ee, np.asarray(re), rng)
            rows.append({
                "Cancer": DISP[cancer], "Model": ALAB[arm], "Selected gene-variant features, n": nf,
                "MSK held-out C": f"{im:.3f} ({ilo:.3f}-{ihi:.3f})",
                "GENIE BPC C": f"{em:.3f} ({elo:.3f}-{ehi:.3f})",
                "Transfer dC": round(em - im, 3),
            })
            comb.append({"cancer": cancer, "arm": arm, "nf_internal": nf,
                         "C_internal": im, "nf_msk": nf, "C_external": em,
                         "internal_to_external_drop": round(im - em, 3)})
            print(f"{DISP[cancer]:5} {ALAB[arm]:26} nf={nf:3} MSK={im:.3f} GENIE={em:.3f}", flush=True)

        # paired bootstrap ΔC (compact − matched-size gene) on the SAME resampled MSK test patients
        rc = np.asarray(risks["compact_fixed"][0]); rg = np.asarray(risks["gene_capped_K"][0])
        n = len(ti); d = []
        for _ in range(B):
            bi = rng.integers(0, n, n)
            if ei[bi].sum() < 2:
                continue
            d.append(concordance_index(ti[bi], -rc[bi], ei[bi]) - concordance_index(ti[bi], -rg[bi], ei[bi]))
        d = np.array(d)
        paired_rows.append({"Cancer": DISP[cancer],
                            "Paired ΔC (compact − matched-size gene)": f"{d.mean():+.3f} "
                            f"({np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f})"})
        print(f"{DISP[cancer]:5} paired ΔC (compact-gene) = {d.mean():+.3f}", flush=True)

    # back up originals once, then overwrite
    for name in ("benchmark_bootstrap_table.csv", "combined_internal_external.csv"):
        src = OUT / name; bak = OUT / name.replace(".csv", ".prepanel.csv")
        if src.exists() and not bak.exists():
            shutil.copy(src, bak)
    pd.DataFrame(rows).to_csv(OUT / "benchmark_bootstrap_table.csv", index=False)
    pd.DataFrame(comb).to_csv(OUT / "combined_internal_external.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(OUT / "panel_coverage_harmonization.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_delta_compact_vs_gene.csv", index=False)
    print("\n" + pd.DataFrame(cov_rows).to_string(index=False))
    print(f"\nwrote {OUT}/(benchmark_bootstrap_table, combined_internal_external, panel_coverage_harmonization).csv")


if __name__ == "__main__":
    main()
