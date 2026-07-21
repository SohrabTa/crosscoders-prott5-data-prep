"""Per-layer decoder-norm profiles of concept-paired crosscoder features,
colored by the biological concept *category* the feature is associated with.

Replicates the original sparse-crosscoder per-layer-norm figure, but recolors
the lines by biology (Domain / Region / Disulfide bond / ...) instead of peak layer.

x-axis: ProtT5 encoder layer (1..24)
y-axis: feature decoder norm at each layer, per-feature max-normalized to [0,1]
color : concept category of the feature's best (per-domain-F1) pairing
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path("/Users/sohrab.tawana/private/crosscoder")

# Variant selector. auxfix = the AuxK-fix re-run (the hand-in, 61 concepts / 409 features);
# pre-auxfix = the superseded original run (51 concepts / 219 features), kept reproducible.
# Override with VARIANT=pre-auxfix in the environment.
VARIANT = os.environ.get("VARIANT", "auxfix")

_PATHS = {
    "pre-auxfix": dict(
        ckpt="model_checkpoints/crosscoder_l8192_k32_bs512_full_2026-03-12_06-03-41/crashed_epoch_0_step_2519836/model.pt",
        pairings="data/crosscoder_eval/pre-auxfix/real/uniprotkb_modern_score45_67k/test_counts/heldout_all_top_pairings.csv",
        suffix="",
    ),
    "auxfix": dict(
        ckpt="model_checkpoints/crosscoder_l8192_k32_bs512_full_auxfix_2026-06-06_07-04-40/final_epoch_0_step_2519836/model.pt",
        pairings="data/crosscoder_eval/auxfix/real/uniprotkb_modern_score45_67k/test_counts/heldout_all_top_pairings.csv",
        suffix="_auxfix",
    ),
}
_cfg = _PATHS[VARIANT]
CKPT = ROOT / _cfg["ckpt"]
PAIRINGS = ROOT / _cfg["pairings"]
OUTDIR = ROOT / "data/figures"
OUT = OUTDIR / f"layerwise_feature_anatomy{_cfg['suffix']}.png"
OUT_FACET = OUTDIR / f"layerwise_feature_anatomy_faceted{_cfg['suffix']}.png"
OUT_FACET_FAMILY = OUTDIR / f"layerwise_feature_anatomy_faceted_family{_cfg['suffix']}.png"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Enzyme-family / functional display grouping (Scheme C) -----------------
# Splits the two opaque super-buckets (Domain 289, Region 78) by enzyme family;
# keeps the biologically-distinct non-enzyme categories (Zinc finger, Coiled
# coil, Disulfide bond) as their own panels; folds the enzyme-machinery Motif
# (Q motif -> ATPase/helicase) in with the enzymes; pools the n<=2 PTM/targeting
# singletons into one panel. Motivation + balance metrics:
# documentation/scripts/category_scheme_balance.py. Boundary calls to eyeball:
# Motif_Q motif -> ATPase; Zn fingers kept separate from DNA/RNA-binding domains
# (merge candidate); Globin/C-lectin/NR-LBD are ligand-binders lumped into the
# protein-interaction "Interaction module"; Region_N-domain -> Structural.
FAMILY_DISPLAY = {
    # enzymes (from Domain + Region + the enzyme-machinery Motif)
    "Domain_Protein kinase": "Kinase/phosphatase",
    "Region_Ribokinase": "Kinase/phosphatase",
    "Domain_Tyrosine-protein phosphatase": "Kinase/phosphatase",
    "Domain_Peptidase S1": "Peptidase",
    "Domain_Peptidase A1": "Peptidase",
    "Domain_Peptidase M12B": "Peptidase",
    "Region_Pyrophosphorylase": "Transferase",
    "Region_N-acetyltransferase": "Transferase",
    "Domain_N-acetyltransferase": "Transferase",
    "Region_Cytidylyltransferase": "Transferase",
    "Domain_UBC core": "Transferase",
    "Domain_GST C-terminal": "Transferase",
    "Region_Uroporphyrinogen-III C-methyltransferase": "Transferase",
    "Domain_Radical SAM core": "Redox/cofactor enzyme",
    "Domain_Thioredoxin": "Redox/cofactor enzyme",
    "Domain_KARI N-terminal Rossmann": "Redox/cofactor enzyme",
    "Domain_KARI C-terminal knotted": "Redox/cofactor enzyme",
    "Domain_Rieske": "Redox/cofactor enzyme",
    "Domain_FAD-binding FR-type": "Redox/cofactor enzyme",
    "Region_Precorrin-2 dehydrogenase /sirohydrochlorin ferrochelatase": "Redox/cofactor enzyme",
    "Domain_HD": "Other enzyme",
    "Domain_Nudix hydrolase": "Other enzyme",
    "Domain_Glutamine amidotransferase type-1": "Other enzyme",
    "Domain_Rhodanese": "Other enzyme",
    "Region_CPSase": "Other enzyme",
    "Domain_AB hydrolase-1": "Other enzyme",
    "Domain_PPIase cyclophilin-type": "Other enzyme",
    "Domain_Response regulatory": "ATPase/GTPase/signaling",
    "Domain_Helicase ATP-binding": "ATPase/GTPase/signaling",
    "Domain_Helicase C-terminal": "ATPase/GTPase/signaling",
    "Domain_G-alpha": "ATPase/GTPase/signaling",
    "Motif_Q motif": "ATPase/GTPase/signaling",
    # binding / interaction modules
    "Domain_F-box": "Interaction module",
    "Domain_Globin": "Interaction module",
    "Domain_PDZ": "Interaction module",
    "Domain_C-type lectin": "Interaction module",
    "Domain_SH3": "Interaction module",
    "Domain_J": "Interaction module",
    "Domain_NR LBD": "Interaction module",
    "Domain_B30.2/SPRY": "Interaction module",
    "Region_I-domain": "Interaction module",
    "Domain_THUMP": "DNA/RNA-binding domain",
    "Domain_bHLH": "DNA/RNA-binding domain",
    "Domain_Sm": "DNA/RNA-binding domain",
    "Zinc finger_any": "Zinc finger",
    "Zinc finger_RING-type": "Zinc finger",
    "Zinc finger_NR C4-type": "Zinc finger",
    # structural / disorder / PTM (non-enzyme, kept distinct)
    "Coiled coil": "Coiled coil",
    "Domain_Collagen-like": "Structural domain",
    "Domain_IF rod": "Structural domain",
    "Region_N-domain": "Structural domain",
    "Region_Disordered": "Disordered/low-complexity",
    "Compositional bias_Acidic residues": "Disordered/low-complexity",
    "Disulfide bond": "Disulfide bond",
    "Glycosylation_N-linked (GlcNAc...) asparagine": "Other PTM/targeting",
    "Modified residue_N6-(pyridoxal phosphate)lysine": "Other PTM/targeting",
    "Transit peptide_any": "Other PTM/targeting",
}


def decoder_norms_per_layer(state, unfold_scale=True):
    """[n_latents, n_layers] scale-invariant decoder norms."""
    w_dec = state["_W_dec_LXoDo"][:, 0, :, :]          # [L=8192, P=24, D=1024]
    norms = w_dec.norm(dim=-1)                          # [8192, 24]
    if unfold_scale and bool(state["is_folded"]):
        scale_P = state["folded_scaling_factors_out_Xo"][0]   # [24]
        norms = norms * scale_P[None, :]
    return norms


def best_pairing_per_feature(df):
    """One category per feature: the pairing with the highest per-domain F1."""
    idx = df.groupby("feature")["f1_per_domain"].idxmax()
    best = df.loc[idx, ["feature", "concept", "f1_per_domain"]].copy()
    best["category"] = best["concept"].str.split("_").str[0]
    best["family"] = best["concept"].map(FAMILY_DISPLAY)
    return best


def main():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    norms = decoder_norms_per_layer(state)             # [8192, 24]
    norms = norms / norms.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    norms = norms.numpy()
    n_layers = norms.shape[1]

    pairings = pd.read_csv(PAIRINGS)
    best = best_pairing_per_feature(pairings)

    counts = best["category"].value_counts()
    order = counts.index.tolist()                       # largest first
    cmap = plt.get_cmap("tab10")
    colors = {cat: cmap(i % 10) for i, cat in enumerate(order)}
    x = list(range(1, n_layers + 1))

    def feats(cat):
        return [int(f) for f in best.loc[best["category"] == cat, "feature"]]

    SAMPLE_PER_CAT = 20     # per-panel line-subsample cap for the faceted figures
    RNG_SEED = 0

    def render_facet(group_col, group_order, outpath, cmap_name="viridis"):
        """Small multiples, one panel per group in group_order: up to
        SAMPLE_PER_CAT randomly-subsampled per-feature lines colored by per-domain
        F1 ([0.5,1.0] viridis), bold black mean over ALL features, shared colorbar,
        two-role legend in the first empty slot. Used for both the Swiss-Prot
        category facet and the enzyme-family facet so their styling stays identical."""
        rng = np.random.default_rng(RNG_SEED)
        fcmap = plt.get_cmap(cmap_name)
        fnorm = Normalize(0.5, 1.0)
        ncol = 4
        nrow = (len(group_order) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow),
                                 sharex=True, sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for i, g in enumerate(group_order):
            a = axes[i]
            sub = best[best[group_col] == g]
            all_ids = [int(f) for f in sub["feature"]]
            shown = sub.iloc[rng.choice(len(sub), SAMPLE_PER_CAT, replace=False)] \
                if len(sub) > SAMPLE_PER_CAT else sub
            for fid, f1 in zip(shown["feature"].astype(int), shown["f1_per_domain"]):
                a.plot(x, norms[fid], color=fcmap(fnorm(f1)), alpha=0.85, linewidth=1.1)
            if len(all_ids) >= 3:                          # mean over ALL, not the subsample
                a.plot(x, norms[all_ids].mean(0), color="black", linewidth=2.4)
            label = f"{g}  (n={len(sub)})" if len(sub) <= SAMPLE_PER_CAT \
                else f"{g}  (n={len(sub)}, {SAMPLE_PER_CAT} shown)"
            a.set_title(label, fontsize=10)
            a.set_xlim(1, n_layers); a.set_ylim(0, 1.05); a.set_xticks([1, 12, 24])
            a.spines[["top", "right"]].set_visible(False)
            a.tick_params(labelbottom=True)                # every panel keeps its own x labels
        for j in range(len(group_order), len(axes)):
            axes[j].axis("off")
        if len(group_order) < len(axes):                   # two-role legend in first empty slot
            axes[len(group_order)].legend(
                handles=[
                    Line2D([0], [0], color="black", lw=2.4, label="Category mean (all features)"),
                    Line2D([0], [0], color=fcmap(fnorm(0.8)), lw=1.2,
                           label="Individual feature (color = F1)"),
                ],
                loc="center", frameon=False, fontsize=11)
        fig.supxlabel("ProtT5 encoder layer", fontsize=12)
        fig.supylabel("Decoder norm (per-feature normalized)", fontsize=12)
        sm = ScalarMappable(norm=fnorm, cmap=fcmap); sm.set_array([])
        fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01).set_label(
            "per-domain F1 (pairing confidence)")
        fig.savefig(outpath, dpi=200)
        plt.close(fig)

    # ---- Figure 1: combined panel — per-category mean +/- 1 SEM band ----
    # Mean decoder-norm profile per Swiss-Prot category (n>=3); shaded band = +/-1
    # standard error of the mean (SD/sqrt(n)) at each layer, i.e. the uncertainty of
    # the plotted mean. SEM (not SD) keeps 6 overlapping categories legible: large
    # categories get tight bands, small-n ones are honestly wider. The full
    # per-feature spread is shown separately in the appendix facet.
    drawn = [c for c in order if len(feats(c)) >= 3]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for cat in drawn:
        ids = feats(cat)
        m = norms[ids].mean(0)
        sem = norms[ids].std(0) / (len(ids) ** 0.5)
        ax.fill_between(x, np.clip(m - sem, 0, 1), np.clip(m + sem, 0, 1),
                        color=colors[cat], alpha=0.15, linewidth=0)
    for cat in drawn:                                      # means on top of every band
        ax.plot(x, norms[feats(cat)].mean(0), color=colors[cat], linewidth=2.6)
    handles = [plt.Line2D([0], [0], color=colors[c], lw=2.6,
                          label=f"{c} (n={counts[c]})") for c in drawn]
    ax.legend(handles=handles, title="Concept category (mean $\\pm$ 1 SEM)", fontsize=9,
              title_fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xlabel("ProtT5 encoder layer")
    ax.set_ylabel("Decoder norm (per-feature normalized)")
    ax.set_xlim(1, n_layers); ax.set_ylim(0, 1.05); ax.set_xticks([1, 6, 12, 18, 24])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: appendix detail — Swiss-Prot category facet (Figure 1's categories) ----
    render_facet("category", drawn, OUT_FACET)

    # ---- Figure 3: enzyme-family facet — splits Domain+Region by enzyme family ----
    # Same styling as Figure 2, grouped by FAMILY_DISPLAY (Scheme C) instead of the
    # Swiss-Prot type. "Other PTM/targeting" is dropped (glyco+modres+transit pooled
    # = a meaningless mean). Kept available; the paper uses the Figure 2 facet.
    if best["family"].isna().any():
        missing = sorted(best.loc[best["family"].isna(), "concept"].unique())
        raise ValueError(f"FAMILY_DISPLAY missing concepts: {missing}")
    DROP_FAMILIES = {"Other PTM/targeting"}
    fam_counts = best["family"].value_counts()                 # largest first
    fam_order = [f for f in fam_counts.index if f not in DROP_FAMILIES]
    render_facet("family", fam_order, OUT_FACET_FAMILY)

    print(f"wrote {OUT}\nwrote {OUT_FACET}\nwrote {OUT_FACET_FAMILY}")
    print(f"facets: <= {SAMPLE_PER_CAT} lines/panel subsampled (seed {RNG_SEED}), mean over all; "
          f"dropped {sorted(DROP_FAMILIES)} from family facet")
    print(f"variant={VARIANT}  {len(best)} concept-paired features across {len(order)} categories")
    print(counts.to_string())

    # Per-category peak layer: argmax of the per-category mean normalized decoder-norm
    # profile (only where n >= 3, matching the bold mean line). Anchors the outline's
    # qualitative "where each concept category lives" claims to computed output.
    print("\nper-category peak layer (argmax of mean profile, n>=3):")
    rows = []
    for cat in order:
        ids = feats(cat)
        if len(ids) < 3:
            continue
        mean_profile = norms[ids].mean(0)
        peak = int(mean_profile.argmax()) + 1          # 1-indexed layer
        rows.append((cat, int(counts[cat]), peak))
    for cat, n, peak in sorted(rows, key=lambda r: r[2]):
        print(f"  {cat:<20} n={n:<4} peak≈layer {peak}")


if __name__ == "__main__":
    main()
