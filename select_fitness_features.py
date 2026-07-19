"""Select the crosscoder features that track ProteinGym DMS fitness best, per assay.

Reads the auxfix full-feature Spearman scan (per-(assay, feature) |pool_mean_abs|
correlation) and, for each assay, ranks features by |rho| and keeps the top-K. The
output feeds the LLM-autointerp scoping question: are the top-5-per-assay trackers all
high-|rho| (worth labelling all of them) or does |rho| drop off after rank 1 (top-1
suffices)? Also tags each selected feature paired/unpaired (vs the InterPLM pairings)
and records its firing density, to later test the "dense composition feature" confound.

Inputs (auxfix only):
  data/proteingym/full_feature_spearman_auxfix/full_feature_spearman.csv
      cols: DMS_id, feature, n_variants, fire_rate_mean, pool_mean_abs_sp, delta_at_pos_sp
  data/crosscoder_eval/auxfix/real/uniprotkb_modern_score45_67k/test_counts/
      heldout_all_top_pairings.csv   (feature-id column -> the paired/labelled set)

Outputs:
  data/proteingym/fitness_top_features_auxfix.csv   (assay, rank, feature, abs_rho, ...)
  data/proteingym/fitness_top_features_auxfix_unique.csv  (deduped feature list for autointerp)
  + a per-rank |rho| distribution printed to stdout (the top-1-vs-top-5 decision).

Deterministic (stable sort, no RNG).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
FF = ROOT / "data/proteingym/full_feature_spearman_auxfix/full_feature_spearman.csv"
PAIRINGS = ROOT / "data/crosscoder_eval/auxfix/real/uniprotkb_modern_score45_67k/test_counts/heldout_all_top_pairings.csv"
OUT = ROOT / "data/proteingym/fitness_top_features_auxfix.csv"
OUT_UNIQUE = ROOT / "data/proteingym/fitness_top_features_auxfix_unique.csv"
TOPK = 5
METRIC = "pool_mean_abs_sp"  # the headline readout


def main():
    df = pd.read_csv(FF)
    df = df.dropna(subset=[METRIC]).copy()
    df["abs_rho"] = df[METRIC].abs()

    paired_ids = set(pd.read_csv(PAIRINGS)["feature"].astype(int).unique())
    print(f"{len(paired_ids)} paired (InterPLM-labelled) features; "
          f"{df['DMS_id'].nunique()} assays; {len(df):,} (assay,feature) rows")

    # Per assay: rank by |rho| desc (stable), keep top-K.
    df = df.sort_values(["DMS_id", "abs_rho"], ascending=[True, False], kind="stable")
    df["rank"] = df.groupby("DMS_id").cumcount() + 1
    top = df[df["rank"] <= TOPK].copy()
    top["paired"] = top["feature"].astype(int).isin(paired_ids)

    # --- the decision: per-rank |rho| distribution across assays ---
    print(f"\nPer-rank |rho| across assays (metric={METRIC}, top-{TOPK} per assay):")
    print(f"{'rank':>4} {'median':>7} {'q25':>6} {'q75':>6} {'min':>6} {'%unpaired':>10}")
    for r in range(1, TOPK + 1):
        sub = top[top["rank"] == r]
        pct_unpaired = 100 * (~sub["paired"]).mean()
        print(f"{r:>4} {sub['abs_rho'].median():>7.3f} {sub['abs_rho'].quantile(.25):>6.3f} "
              f"{sub['abs_rho'].quantile(.75):>6.3f} {sub['abs_rho'].min():>6.3f} {pct_unpaired:>9.0f}%")

    # unique feature sets
    uniq_top1 = top.loc[top["rank"] == 1, "feature"].astype(int).nunique()
    uniq_topk = top["feature"].astype(int).nunique()
    print(f"\nunique features: top-1 per assay = {uniq_top1}; top-{TOPK} per assay = {uniq_topk}")
    print(f"of the top-{TOPK} unique set, unpaired = "
          f"{top.groupby('feature')['paired'].first().pipe(lambda s: (~s).sum())}")

    # firing density of the selected (dense-feature confound check)
    fr = top.groupby("feature")["fire_rate_mean"].first()
    print(f"selected-feature fire_rate_mean: median {fr.median():.3f}, "
          f"q75 {fr.quantile(.75):.3f}, max {fr.max():.3f}")

    top.to_csv(OUT, index=False)
    # deduped, ranked by how often a feature is a top tracker + its best |rho|
    agg = (top.groupby("feature")
              .agg(n_assays_topk=("DMS_id", "nunique"),
                   best_abs_rho=("abs_rho", "max"),
                   median_abs_rho=("abs_rho", "median"),
                   fire_rate_mean=("fire_rate_mean", "first"),
                   paired=("paired", "first"))
              .sort_values(["n_assays_topk", "best_abs_rho"], ascending=False)
              .reset_index())
    agg.to_csv(OUT_UNIQUE, index=False)
    print(f"\nwrote {OUT}\nwrote {OUT_UNIQUE} ({len(agg)} unique features)")


if __name__ == "__main__":
    main()
