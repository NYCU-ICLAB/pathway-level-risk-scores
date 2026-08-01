# Gene-level vs pathway-level benchmark — internal (MSK-CHORD held-out test set)

Supports **Figure 6a** (parsimony–performance comparison). Benchmarks the compact
pathway-level score against unaggregated gene-level penalized-Cox models on the same
MSK-CHORD held-out test set.

## Design
- Fixed 8:2 split (`random_state=42`, stratified on `Event_OS`); SAMPLE_ID is
  recovered per row so that gene-level events and survival are matched 1:1.
- **Genomic-only comparison**: the compact pathway score uses no clinical covariates,
  so the gene-level arms add none either (adding clinical covariates would favour the
  higher-dimensional gene-level models).
- All arms evaluated on the same test set and metrics: Harrell C, Uno C-IPCW,
  time-dependent AUC@24m, Accuracy@24m.
- Uncertainty: test-set bootstrap (B = 1000) for 95% CIs, plus paired ΔC and two-sided
  bootstrap p-values against the deployed compact model.

## Arms (per cancer)
| arm | description | n features |
|---|---|---|
| `compact_fixed` | final 4–5 pathway features with the published fixed β (deployed model) | 4–5 |
| `compact_refit` | same 4–5 pathway features, CoxPH refit on train | 4–5 |
| `gene_restricted` | gene-event Coxnet; candidates = retained-pathway genes (≥1% prevalence) | 8–54 |
| `gene_unrestricted` | gene-event Coxnet; candidates = all recurrently altered panel genes | 11–107 |
| `gene_capped_K` | `gene_restricted` forced to ≈K features (matched parsimony) | 2–5 |

## Findings
1. Deployed fixed-β compact ≈ refit compact (ΔC ≈ 0, p > 0.3 in all five cancers):
   fixing coefficients without refitting incurs no discrimination loss.
2. Gene-level models gain only marginal discrimination and only when using ~2 to >20×
   more features (ΔC ≈ +0.01 to +0.06; significant only for `gene_unrestricted` in
   BRCA/LUAD/PRAD internally).
3. At matched parsimony (K = 4–5), the pathway compact score is ≥ gene-level in all
   five cancers (`gene_capped_K`).

Internal superiority of large gene-level models is not robust externally
(see `../external/`); the overall conclusion is **comparable discrimination with fewer,
portable, interpretable features**, not pathway superiority.

> Note: Accuracy@24m is threshold-sensitive under low event rates; discrimination is
> read primarily from C-index / C-IPCW.

## Files
- `headtohead_metrics.csv` — 5 cancers × 5 arms × 4 metrics + feature counts
- `headtohead_bootstrap_deltaC.csv` — ΔC vs `compact_fixed` + 95% CI + p per arm
- `panelA_parsimony_frontier.{png,pdf}` — C-index vs number of features
- `panelB_equal_parsimony.{png,pdf}` — compact vs gene at matched K
- `panelC_forest_deltaC.{png,pdf}` — ΔC forest plot

## Reproduce
```
python3 ../../scripts/gene_vs_pathway_headtohead.py --boot 1000
python3 ../../scripts/plot_gene_vs_pathway_headtohead.py
```
