#!/usr/bin/env python3
"""Figures for the gene-vs-pathway head-to-head (genomic-only).

Panel A: parsimony frontier — test C-index vs #features per arm, per cancer.
         Shows the compact pathway score sitting at far-left (few features) with
         C-index competitive with gene-level models that need 7-25x more features.
Panel B: equal-parsimony bar — compact_fixed vs gene_capped_K (same K) C-index.
Panel C: forest of DeltaC (compact_fixed - gene arm) with bootstrap 95% CI.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "figures_tables" / "gene_vs_pathway_headtohead"
try:
    import sys; sys.path.insert(0, str(ROOT / "scripts"))
    from figure_style import apply_arial_style
    apply_arial_style()
except Exception:
    pass

CANCERS = ["BRCA", "LUAD", "PAAD", "PRAD", "CRC"]
ARM_COLOR = {"compact_fixed": "#1a5276", "compact_refit": "#5499c7",
             "gene_restricted": "#e59866", "gene_unrestricted": "#cb4335",
             "gene_capped_K": "#7d3c98"}
ARM_LABEL = {"compact_fixed": "Compact (fixed β, deployed)",
             "compact_refit": "Compact (refit)",
             "gene_restricted": "Gene-level (pathway genes)",
             "gene_unrestricted": "Gene-level (all panel genes)",
             "gene_capped_K": "Gene-level (capped to K)"}

m = pd.read_csv(D / "headtohead_metrics.csv")
b = pd.read_csv(D / "headtohead_bootstrap_deltaC.csv")

# ---------- Panel A: parsimony frontier ----------
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=False)
for ax, c in zip(axes, CANCERS):
    sub = m[m.cancer == c]
    for _, r in sub.iterrows():
        ax.scatter(r.n_features, r.c_index, s=70, color=ARM_COLOR[r.arm],
                   zorder=3, edgecolor="white", linewidth=0.6)
    cf = sub[sub.arm == "compact_fixed"].iloc[0]
    ax.axhline(cf.c_index, ls="--", lw=0.8, color="#1a5276", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_title(c, fontsize=11, weight="bold")
    ax.set_xlabel("# features (log)", fontsize=8)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Test C-index", fontsize=9)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=ARM_COLOR[a],
                      label=ARM_LABEL[a], markersize=8) for a in ARM_COLOR]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Parsimony frontier: pathway compact score matches gene-level "
             "discrimination at a fraction of the feature count (genomic-only)",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig(D / "panelA_parsimony_frontier.png", dpi=200, bbox_inches="tight")
fig.savefig(D / "panelA_parsimony_frontier.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Panel B: equal-parsimony bar ----------
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(CANCERS)); w = 0.38
comp = [m[(m.cancer == c) & (m.arm == "compact_fixed")].c_index.iloc[0] for c in CANCERS]
capk = [m[(m.cancer == c) & (m.arm == "gene_capped_K")].c_index.iloc[0] for c in CANCERS]
kfe = [int(m[(m.cancer == c) & (m.arm == "compact_fixed")].n_features.iloc[0]) for c in CANCERS]
ax.bar(x - w/2, comp, w, label="Pathway compact (fixed β)", color="#1a5276")
ax.bar(x + w/2, capk, w, label="Gene-level, same K", color="#7d3c98")
ax.axhline(0.5, ls=":", color="gray", lw=0.8)
for i, k in enumerate(kfe):
    ax.text(i, max(comp[i], capk[i]) + 0.008, f"K={k}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(CANCERS)
ax.set_ylabel("Test C-index"); ax.set_ylim(0.5, 0.72)
ax.set_title("Equal parsimony (K=4–5): pathway aggregation ≥ gene-level in every cancer")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(D / "panelB_equal_parsimony.png", dpi=200)
fig.savefig(D / "panelB_equal_parsimony.pdf")
plt.close(fig)

# ---------- Panel C: forest of DeltaC ----------
arms = ["gene_capped_K", "gene_restricted", "gene_unrestricted"]
fig, ax = plt.subplots(figsize=(8.5, 6))
ypos, ylabels = [], []
y = 0
for c in CANCERS:
    for a in arms:
        r = b[(b.cancer == c) & (b.comparison == f"compact_fixed_minus_{a}")]
        if r.empty:
            continue
        r = r.iloc[0]
        col = "#2874a6" if r.delta_c_mean > 0 else "#c0392b"
        ax.plot([r.delta_c_ci_lo, r.delta_c_ci_hi], [y, y], color=col, lw=2)
        ax.scatter(r.delta_c_mean, y, color=col, zorder=3, s=40)
        star = " *" if r.boot_p_two_sided < 0.05 else ""
        ylabels.append(f"{c} — vs {ARM_LABEL[a].split('(')[1][:-1]}{star}")
        ypos.append(y); y += 1
    y += 0.6
ax.axvline(0, ls="--", color="black", lw=0.8)
ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlabel("ΔC = C(pathway compact) − C(gene-level)   [>0 favors pathway]")
ax.set_title("Paired test-set bootstrap ΔC (1000 resamples); * p<0.05")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(D / "panelC_forest_deltaC.png", dpi=200)
fig.savefig(D / "panelC_forest_deltaC.pdf")
plt.close(fig)
print(f"wrote panels A/B/C to {D}")
