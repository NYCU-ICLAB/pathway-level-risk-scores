# Constituent-gene decomposition of the compact features

Supports **Figure 7**. Decomposes each cancer's final 4–5 compact features into their
constituent genes.

For each feature we report:
- **Structure**: the mapped pathway, the alteration types used (mut / amp / del / sv),
  and the aggregation logic.
- **Member genes**: TCGA 10-pathway template genes restricted to those actually altered
  in that cancer's training cohort.
- **Per-gene weight**: training-cohort alteration frequency, univariate OS Cox HR and p,
  and `contribution = frequency × |log HR|` (ranked within the feature).

Contribution ranking indicates which genes drive each feature. Alignment matches the
head-to-head benchmark (SAMPLE_ID recovered so gene events and survival are matched 1:1).

## Representative results (`feature_to_gene_summary.csv`)
- **BRCA** `TP53_mut_rate` ← almost entirely TP53 (f=0.45, HR 1.67); ATM/CHEK2/MDM4 minor.
- **BRCA** `All_Pathway_Alteration_Count` ← TP53, ERBB2 (HR 0.75, protective), MYC, FGFR1.
- **LUAD** `NRF2_zsum` ← KEAP1 (f=0.13, HR 1.66), NFE2L2, CUL3 (canonical NRF2 axis).
- **PAAD** `All_Pathway_Alteration_Count` ← KRAS (f=0.93), TP53 (0.76), CDKN2A.
- **PRAD** `Cell_Cycle_zsum` ← RB1 (HR 2.21), CCND1 (HR 1.78); `WNT_zsum` ← APC/CTNNB1/AMER1.
- **CRC** `RTK_RAS_any_rate` ← KRAS (f=0.44); `DDR_zsum` / `Chromatin_any_rate` mostly HR<1.

## Single-gene dominance
Each feature is classified by `top1_share_pct` (the leading gene's share of total
contribution) into `dominance_class`:

| class | definition | examples (leading-gene share) |
|---|---|---|
| single-gene (≥90%) | essentially one gene | TP53_mut/any_rate (TP53 94–98%), Cell_Cycle_sv_hit (RB1 100%), TGF_Beta_amp_hit (SMAD2 100%) |
| driver-anchored (50–90%) | one dominant driver + minor | LUAD NRF2_zsum (KEAP1 88%), PAAD MYC_any_rate (80%), TGF_Beta_any_rate (SMAD4 75%) |
| multi-gene (<50%) | genuinely polygenic | all five All_Pathway_Alteration_Count (leading gene only 6–28%), Chromatin/DDR/WNT/Cell_Cycle_zsum |

15 of 24 features are genuinely multi-gene, and the always-selected backbone feature
`All_Pathway_Alteration_Count` is multi-gene in all five cancers (leading gene only
6–28%), so the score is not reducible to a single gene such as TP53.

## Two complementary views
1. **Frequency-weighted** `contribution = frequency × |log HR|`
   (`<CANCER>_feature_to_gene.png`): which genes drive the feature in practice.
2. **Unweighted** `|log HR|` (`<CANCER>_feature_to_gene_byHR.png`, genes with n≥10 only):
   pure prognostic strength, surfacing moderately rare but high-HR genes.

## Files
- `feature_to_gene_long.csv` — one row per (cancer, feature, gene): pathway, alteration
  type, frequency, HR, p, `abs_logHR`, contribution, within-feature rank.
- `feature_to_gene_summary.csv` — one row per feature: pathway, alteration type, member
  gene count, `top1_gene`, `top1_share_pct`, `dominance_class`, top genes by each view.
- `<CANCER>_feature_to_gene.{png,pdf}` — frequency-weighted contribution bars.
- `<CANCER>_feature_to_gene_byHR.{png,pdf}` — unweighted |log HR| bars (n≥10).
- Colours: red = HR≥1 (worse prognosis), blue = HR<1 (better/protective).

## Reproduce
```
python3 ../scripts/feature_to_gene_traceback.py
```
