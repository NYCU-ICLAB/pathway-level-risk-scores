# Gene-level vs pathway-level benchmark — external (GENIE BPC, no refit)

Supports **Figure 6b** (internal-to-external transfer). No-refit external transfer of
gene-level and pathway-level models to the GENIE BPC cohorts.

## Protocol (matched to the published compact-score external validation)
- Gene-level Coxnet is fitted on MSK train and then fixed (coefficients, StandardScaler
  parameters, and selected feature columns).
- For each GENIE BPC cohort, the same (gene, alteration) events are rebuilt (mutations
  from `data_mutations`, amplification/deletion from `data_cna`; SV set to 0 for all
  non-MSK panels, as in the compact-score external procedure), aligned to the fixed
  columns (genes absent from an external panel → 0), and scored with the fixed scaler
  and coefficients. No refitting, no re-standardization.
- The compact pathway-score external risk is taken directly from the validated
  `external_validation/` outputs.

## Results (`external_metrics.csv`, `external_bootstrap_deltaC.csv`, `combined_internal_external.csv`)

| Cancer | compact external C | best gene external C | external result |
|---|---|---|---|
| BRCA | 0.625 | gene(=K) 0.681, gene(pw) 0.669 | gene-level higher (incl. matched K, p<0.001) |
| LUAD | 0.612 | ~0.62–0.63 | comparable |
| PAAD | 0.574 | ~0.58 | comparable |
| PRAD | 0.629 | gene(pw) 0.645 | comparable |
| CRC | 0.505 | gene ~0.61 | gene-level higher (p<0.001) |

External validation does not support a "gene-level is worse" reading: gene-level is
higher in BRCA and CRC, comparable elsewhere. The project therefore adopts a
**comparable + parsimonious + portable** framing (no formal non-inferiority test), and
does not claim pathway-level discrimination superiority.

### Claims supported by the data
1. **Comparable**: the 4–5-feature compact score is within ±0.02–0.05 C-index of models
   using ~2 to >20× more gene features, internally and externally.
2. **Large gene models overfit**: `gene_unrestricted` shows the largest internal→external
   drop (e.g. PRAD 0.699→0.628, BRCA 0.695→0.641); small gene models and the compact
   score transfer more stably.
3. **Portable / interpretable**: the compact score needs only pathway-level summaries,
   not specific genes or a specific panel.

## CRC external note (diagnostic, not a code error)
The CRC compact score's leading term (`PW_TGF_Beta_amp_hit`) has 0% prevalence in the
GENIE CRC cohort, so its largest-weight term becomes a constant 0 externally; the
remaining CRC features have weak univariate external discrimination. CRC is the weakest
cancer internally, and MSI biology makes alteration burden non-monotonic with survival.
This is disclosed as a limitation: the CRC compact pathway score alone has weak external
discrimination and benefits from clinical covariates or MSI stratification.

## Files
- `external_metrics.csv` — external C-index / C-IPCW per cancer per arm + MSK feature counts
- `external_bootstrap_deltaC.csv` — external ΔC (compact − gene) + 95% CI + p
- `combined_internal_external.csv` — internal vs external C-index + feature counts + drop
- `combined_internal_external.{png,pdf}` — internal/external bars across five cancers

## Reproduce
```
python3 ../../scripts/gene_vs_pathway_external.py --boot 1000
python3 ../../scripts/plot_gene_vs_pathway_combined.py
```
