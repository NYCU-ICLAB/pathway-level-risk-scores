#!/usr/bin/env python3
"""
benchmark_bootstrap_table.py  (C8)
==================================
Supplementary benchmark table with bootstrap 95% CIs for the three retained arms
(Pathway compact, Gene(=K), Gene(pathway genes)) — the panel-wide arm is dropped
(coverage). For each cancer and arm: number of selected features, internal
(MSK held-out) C-index with 95% CI, external (GENIE BPC) C-index with 95% CI, and
the transfer ΔC (external − internal). Models are frozen on MSK train and applied
without refitting; CIs are patient-level bootstrap on each evaluation cohort.

Output: figures_tables/gene_vs_pathway_external/benchmark_bootstrap_table.csv
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

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
    rows = []
    for cancer, (raw_dir, out_dir, _onc) in EX.COHORTS.items():
        key = HH.CANCER_ALIAS.get(cancer, cancer)
        K = int((dec["cancer"] == key).sum())
        # internal test cohort
        train, test, tr, te = HH.load_split(cancer)
        ti = test["OS_MONTHS"].to_numpy(float); ei = test["Event_OS"].astype(bool).to_numpy()
        # external cohort
        pw = pd.read_csv(EX.OUT_BASE / out_dir / f"{out_dir}_pathway_features.csv")
        pw = pw[(pw["OS_MONTHS"] > 0) & pw["Event_OS"].notna()].copy()
        sids = pw["SAMPLE_ID"].tolist()
        te_e = pw["OS_MONTHS"].to_numpy(float); ee = pw["Event_OS"].astype(bool).to_numpy()
        genie_ev = EX.build_genie_gene_events(raw_dir, sids)

        gs = HH.retained_gene_set(cancer)
        # frozen models
        from sklearn.preprocessing import StandardScaler
        def frozen(cap):
            ev = HH.build_gene_events(cancer, tr + te, gs)
            prev = ev.loc[tr].mean(0); cols = prev[prev >= HH.PREVALENCE].index.tolist()
            Xtr = ev.loc[tr, cols]; sc = StandardScaler().fit(Xtr.values)
            m, nu = HH.coxnet_select(sc.transform(Xtr.values), HH.make_surv(train), cap_k=cap)
            ri = m.predict(sc.transform(ev.loc[te, cols].values))
            re = EX.apply_frozen({"model": m, "scaler": sc, "cols": cols}, genie_ev)
            return ri, re, nu

        risks = {}
        # compact
        from compact_score.apply_compact_score import apply_score, load_formula
        _, cri = apply_score(test, key, load_formula())
        risks["compact_fixed"] = (cri.to_numpy(), pw["risk_score"].to_numpy(float), K)
        ricap, recap, nucap = frozen(K); risks["gene_capped_K"] = (ricap, recap, nucap)
        rires, reres, nures = frozen(None); risks["gene_restricted"] = (rires, reres, nures)

        for arm in ARMS:
            ri, re, nf = risks[arm]
            im, ilo, ihi = boot_ci(ti, ei, np.asarray(ri), rng)
            em, elo, ehi = boot_ci(te_e, ee, np.asarray(re), rng)
            rows.append({
                "Cancer": DISP[cancer], "Model": ALAB[arm], "Selected features": nf,
                "MSK held-out C": f"{im:.3f} ({ilo:.3f}-{ihi:.3f})",
                "GENIE BPC C": f"{em:.3f} ({elo:.3f}-{ehi:.3f})",
                "Transfer dC": round(em - im, 3),
            })
            print(f"{DISP[cancer]:5} {ALAB[arm]:26} nf={nf:3} MSK={im:.3f} GENIE={em:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "benchmark_bootstrap_table.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUT}/benchmark_bootstrap_table.csv")


if __name__ == "__main__":
    main()
