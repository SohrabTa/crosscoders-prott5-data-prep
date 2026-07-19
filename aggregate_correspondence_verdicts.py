#!/usr/bin/env python3
"""Aggregate the subagent (feature, assay) correspondence verdicts.

Merges verdicts_*.json (written by the judging subagents) with the pair metadata
(pairs.jsonl) and reports the specific/generic/mismatch/vague breakdown overall and
split by paired/unpaired, tracking strength, and assay category. Answers the §5.5
follow-up: do features that track fitness do so via specific matched biology or via
generic structural/compositional priors?

Inputs (read-only):
  - data/llm_autointerp/fitness_auxfix/correspondence/pairs.jsonl
  - data/llm_autointerp/fitness_auxfix/correspondence/verdicts_*.json
Output:
  - data/llm_autointerp/fitness_auxfix/correspondence/correspondence_summary.csv
    (per-pair verdict joined with metadata)
Reproducible; no randomness.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
CORR = ROOT / "data/llm_autointerp/fitness_auxfix/correspondence"
VERDICTS = {"specific_match", "generic_match", "mismatch", "vague"}


def main() -> None:
    pairs = {p["pair_id"]: p for p in
             (json.loads(l) for l in (CORR / "pairs.jsonl").open())}
    verdicts = {}
    for f in sorted(glob.glob(str(CORR / "verdicts_*.json"))):
        for v in json.loads(Path(f).read_text()):
            pid = v["pair_id"]
            verd = str(v.get("verdict", "")).strip()
            if verd not in VERDICTS:
                verd = "vague"  # normalise any off-rubric label
            verdicts[pid] = {"verdict": verd, "reason": v.get("reason", "")}

    missing = sorted(set(pairs) - set(verdicts))
    print(f"pairs: {len(pairs)} | judged: {len(verdicts)} | missing: {len(missing)}")
    if missing:
        print(f"  missing pair_ids (first 20): {missing[:20]}")

    rows = []
    for pid, p in pairs.items():
        if pid not in verdicts:
            continue
        rows.append({
            "pair_id": pid, "feature_id": p["feature_id"], "dms_id": p["dms_id"],
            "paired": p["paired"], "assay_category": p["assay_category"],
            "tracking_abs_rho": p["tracking_abs_rho"], "tracking_rank": p["tracking_rank"],
            "verdict": verdicts[pid]["verdict"], "reason": verdicts[pid]["reason"],
            "feature_label": p["feature_label"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(CORR / "correspondence_summary.csv", index=False)

    def show(title, series):
        vc = series.value_counts()
        n = len(series)
        print(f"\n{title} (n={n})")
        for v in ["specific_match", "generic_match", "mismatch", "vague"]:
            c = int(vc.get(v, 0))
            print(f"  {v:15s} {c:4d}  ({100*c/n:4.1f}%)")

    show("=== overall ===", df["verdict"])
    show("--- unpaired features ---", df[~df.paired]["verdict"])
    show("--- paired features ---", df[df.paired]["verdict"])
    show("--- rank-1 tracker per assay ---", df[df.tracking_rank == 1]["verdict"])
    show("--- strong trackers (abs_rho >= 0.5) ---", df[df.tracking_abs_rho >= 0.5]["verdict"])

    print("\n=== verdict x assay_category (counts) ===")
    print(pd.crosstab(df["assay_category"], df["verdict"]).to_string())
    # per-assay: does any feature specifically match this assay?
    per_assay = df.groupby("dms_id")["verdict"].apply(
        lambda s: "has_specific" if (s == "specific_match").any() else "no_specific")
    print(f"\nassays with >=1 specific_match feature: "
          f"{int((per_assay=='has_specific').sum())}/{per_assay.size}")
    print(f"\nwrote {CORR / 'correspondence_summary.csv'}")


if __name__ == "__main__":
    main()
