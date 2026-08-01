#!/usr/bin/env python3
"""Combined internal vs external figure for the gene-vs-pathway comparison.

Panel: per-cancer grouped bars of internal (MSK test) and external (GENIE BPC)
C-index for each arm, annotated with #features. Communicates the honest,
reframed message: the pathway compact score is NON-INFERIOR to gene-level
using far fewer, portable features; the largest gene models overfit (biggest
internal->external drop). CRC compact fails externally (disclosed limitation).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "figures_tables" / "gene_vs_pathway_external"
try:
    import sys; sys.path.insert(0, str(ROOT / "scripts"))
    from figure_style import apply_arial_style
    apply_arial_style()
except Exception:
    pass

CANCERS = ["BRCA", "LUAD", "PAAD", "PRAD", "CRC"]
ARMS = ["compact_fixed", "gene_capped_K", "gene_restricted", "gene_unrestricted"]
LABEL = {"compact_fixed": "Compact\n(pathway)", "gene_capped_K": "Gene\n(=K)",
         "gene_restricted": "Gene\n(pw genes)", "gene_unrestricted": "Gene\n(all panel)"}
COL_INT = "#5499c7"; COL_EXT = "#1a5276"

m = pd.read_csv(D / "combined_internal_external.csv")

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
for ax, c in zip(axes, CANCERS):
    sub = m[m.cancer == c].set_index("arm")
    x = np.arange(len(ARMS)); w = 0.4
    ci = [sub.loc[a, "C_internal"] if a in sub.index else np.nan for a in ARMS]
    ce = [sub.loc[a, "C_external"] if a in sub.index else np.nan for a in ARMS]
    nf = [int(sub.loc[a, "nf_internal"]) if a in sub.index else 0 for a in ARMS]
    ax.bar(x - w/2, ci, w, label="Internal (MSK test)", color=COL_INT)
    ax.bar(x + w/2, ce, w, label="External (GENIE BPC)", color=COL_EXT)
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    for i, n in enumerate(nf):
        top = np.nanmax([ci[i], ce[i]])
        ax.text(i, top + 0.006, f"{n}f", ha="center", fontsize=7.5)
    ax.set_title(c, fontsize=12, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=7.5)
    ax.set_ylim(0.48, 0.74)
axes[0].set_ylabel("C-index", fontsize=10)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Pathway compact score is non-inferior to gene-level at a fraction of the "
             "features (Nf); large gene models overfit; CRC compact drops externally",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(D / "combined_internal_external.png", dpi=200)
fig.savefig(D / "combined_internal_external.pdf")
print(f"wrote {D}/combined_internal_external.png")
