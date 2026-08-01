#!/usr/bin/env python3
"""
panel_coverage_table.py  (C12)
==============================
Supplementary panel-coverage table for the external gene-level benchmark.

Addresses the reviewer concern that coding a gene as 0 in GENIE could confound
'untested' with 'wild-type'. Because the retained head-to-head arms use only the
cancer-specific retained-pathway gene set (Pathway compact, Gene(=K),
Gene(pathway genes)), this table documents that that gene set is assayed in every
GENIE BPC cohort.

Per cancer it reports, for the pathway-restricted gene model actually applied
externally: number of selected gene-variant features, distinct genes, and how
many of those genes are covered by the GENIE CNA matrix / observed in the GENIE
mutation file / absent from both. (Structural variants are set to 0 for all
non-MSK panels, matching the compact-score external protocol.)

Output: figures_tables/panel_coverage/panel_coverage.csv
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import gene_vs_pathway_external as EX
import gene_vs_pathway_headtohead as HH

DISP = {"BRCA": "IDC", "LUAD": "LUAD", "PAAD": "PAAD", "PRAD": "PRAD", "CRC": "CRC"}
OUT = ROOT / "figures_tables" / "panel_coverage"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cancer, (raw_dir, out_dir, _onc) in EX.COHORTS.items():
        gs = HH.retained_gene_set(cancer)
        fr = EX.fit_frozen_gene_model(cancer, gs)
        used = np.abs(fr["model"].coef_[:, 0]) > 1e-8
        sel_cols = [c for c, u in zip(fr["cols"], used) if u]
        sel_genes = sorted(set(c.split("_")[1] for c in sel_cols))

        mut = pd.read_csv(EX.RAW_BASE / raw_dir / "data_mutations.tsv", sep="\t", low_memory=False)
        cna = pd.read_csv(EX.RAW_BASE / raw_dir / "data_cna.tsv", sep="\t", low_memory=False)
        gmut = set(mut["Hugo_Symbol"]); gcna = set(cna[cna.columns[0]])
        sv_path = EX.RAW_BASE / raw_dir / "data_sv.tsv"

        in_cna = [g for g in sel_genes if g in gcna]
        in_mut = [g for g in sel_genes if g in gmut]
        absent = [g for g in sel_genes if g not in gcna and g not in gmut]
        rows.append({
            "cancer": DISP[cancer], "genie_cohort": out_dir,
            "retained_pathway_genes": len(gs),
            "selected_gene_variant_features": len(sel_cols),
            "distinct_selected_genes": len(sel_genes),
            "selected_genes_in_GENIE_CNA": len(in_cna),
            "selected_genes_in_GENIE_mut": len(in_mut),
            "selected_genes_absent_both": len(absent),
            "absent_gene_list": ";".join(absent) if absent else "",
            "GENIE_CNA_assayed_genes": len(gcna),
            "SV_available": sv_path.exists(),
        })
        print(f"{DISP[cancer]:5} sel_feats={len(sel_cols):3} genes={len(sel_genes):3} "
              f"inCNA={len(in_cna)} inMut={len(in_mut)} absentBoth={len(absent)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel_coverage.csv", index=False)
    print(f"\nwrote {OUT}/panel_coverage.csv")
    print(df[["cancer", "distinct_selected_genes", "selected_genes_in_GENIE_CNA",
              "selected_genes_absent_both", "SV_available"]].to_string(index=False))


if __name__ == "__main__":
    main()
