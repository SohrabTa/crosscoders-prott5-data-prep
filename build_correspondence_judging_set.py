#!/usr/bin/env python3
"""Build the (feature, assay) correspondence-judging set for the LLM-autointerp
follow-up (experiment 06 / ProteinGym negative result, experiment 03).

Question: for each feature that tracks an assay's DMS fitness well, does its LLM
label describe biology relevant to what that assay measures? This assembles every
top-K (feature, assay) pair with the feature's LLM label + the assay's protein and
phenotype, so subagents can judge label<->assay correspondence per pair.

Inputs (read-only):
  - data/proteingym/fitness_top_features_auxfix.csv     (feature x assay top-K pairs)
  - data/external/DMS_substitutions.csv                 (ProteinGym reference: phenotype)
  - data/llm_autointerp/fitness_auxfix/scoring/results_deepseek.parquet  (LLM labels)
Outputs:
  - data/llm_autointerp/fitness_auxfix/correspondence/pairs.jsonl         (all pairs)
  - data/llm_autointerp/fitness_auxfix/correspondence/batch_XX.json       (N batches)
Seed: none (deterministic ordering). Reproducible.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/sohrab.tawana/private/crosscoder")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=15)
    args = ap.parse_args()

    pairs = pd.read_csv(ROOT / "data/proteingym/fitness_top_features_auxfix.csv")
    ref = pd.read_csv(ROOT / "data/external/DMS_substitutions.csv")
    lab = pd.read_parquet(
        ROOT / "data/llm_autointerp/fitness_auxfix/scoring/results_deepseek.parquet"
    )[["feature_id", "summary", "description", "pearson_r"]]

    ref_cols = ["DMS_id", "UniProt_ID", "coarse_selection_type",
                "selection_type", "selection_assay"]
    ref = ref[[c for c in ref_cols if c in ref.columns]].copy()

    # Per-assay-protein UniProt function/keywords/sites (fetch_assay_protein_function.py),
    # so the judge knows what the protein does, not just the coarse selection type.
    fn_path = ROOT / "data/proteingym/assay_protein_function.json"
    prot_fn = json.loads(fn_path.read_text()) if fn_path.exists() else {}

    df = pairs.merge(ref, on="DMS_id", how="left").merge(
        lab, left_on="feature", right_on="feature_id", how="left"
    )
    # A pair is judgeable only if the feature actually got a label (439 scored; a
    # handful of the 454 were dropped in Phase B and have no description).
    n_all = len(df)
    df = df[df["summary"].notna()].reset_index(drop=True)
    print(f"{len(df)} judgeable pairs ({n_all - len(df)} dropped: feature unlabelled)")

    records = []
    for i, r in df.iterrows():
        desc = str(r["description"])
        pf = prot_fn.get(str(r.get("UniProt_ID")), {})
        records.append({
            "pair_id": i,
            "feature_id": int(r["feature"]),
            "dms_id": r["DMS_id"],
            "protein": r.get("UniProt_ID"),
            "protein_name": pf.get("protein_name"),
            "protein_function": pf.get("function"),
            "protein_keywords": pf.get("keywords", []),
            "protein_sites": pf.get("sites", []),
            "assay_category": r.get("coarse_selection_type"),
            "assay_selection_type": (None if pd.isna(r.get("selection_type"))
                                     else r.get("selection_type")),
            "assay_measures": (None if pd.isna(r.get("selection_assay"))
                               else str(r.get("selection_assay"))[:300]),
            "tracking_abs_rho": round(float(r["abs_rho"]), 3),
            "tracking_rank": int(r["rank"]),
            "paired": bool(r["paired"]),
            "label_faithfulness_r": (None if pd.isna(r["pearson_r"])
                                     else round(float(r["pearson_r"]), 3)),
            "feature_label": str(r["summary"]),
            "feature_label_full": desc[:600],
        })

    out_dir = ROOT / "data/llm_autointerp/fitness_auxfix/correspondence"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pairs.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n = args.n_batches
    size = math.ceil(len(records) / n)
    for b in range(n):
        chunk = records[b * size:(b + 1) * size]
        if not chunk:
            continue
        with (out_dir / f"batch_{b:02d}.json").open("w") as f:
            json.dump(chunk, f, indent=1)
    print(f"wrote pairs.jsonl + {min(n, math.ceil(len(records)/size))} batch files "
          f"(~{size} pairs each) to {out_dir}")


if __name__ == "__main__":
    main()
