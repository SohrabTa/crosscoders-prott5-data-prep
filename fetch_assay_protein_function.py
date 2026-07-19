#!/usr/bin/env python3
"""Fetch UniProt function/keywords/sites for the ProteinGym assay target proteins,
so the (feature, assay) correspondence judges know what each protein actually does
(not just the coarse selection type). ProteinGym's UniProt_ID is mostly an entry
NAME (e.g. ACE2_HUMAN), so we query the UniProt search endpoint by `id:`, not the
accessions endpoint.

Inputs (read-only):
  - data/proteingym/fitness_top_features_auxfix.csv   (which assays are in scope)
  - data/external/DMS_substitutions.csv               (DMS_id -> UniProt_ID)
Output:
  - data/proteingym/assay_protein_function.json        (UniProt_ID -> function summary)
Reproducible; hits the public UniProt REST API (needs internet).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path("/Users/sohrab.tawana/private/crosscoder")
SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,id,protein_name,cc_function,keyword,ft_domain,ft_motif,ft_act_site,ft_binding"
BATCH = 40
SITE_TYPES = {"Domain", "Motif", "Active site", "Binding site"}


def summarize(entry: dict) -> dict:
    pd_ = entry.get("proteinDescription", {}) or {}
    pn = ((pd_.get("recommendedName", {}) or {}).get("fullName", {}) or {}).get("value")
    if not pn:  # TrEMBL/unreviewed entries carry submittedName, not recommendedName
        subs = pd_.get("submissionNames") or pd_.get("submittedName") or []
        if isinstance(subs, list) and subs:
            pn = ((subs[0].get("fullName", {}) or {}).get("value"))
    fn = [c for c in entry.get("comments", []) if c.get("commentType") == "FUNCTION"]
    fn_text = fn[0]["texts"][0]["value"] if (fn and fn[0].get("texts")) else None
    kws = [k["name"] for k in entry.get("keywords", [])]
    sites = []
    for ft in entry.get("features", []):
        if ft.get("type") in SITE_TYPES:
            d = ft.get("description") or ft.get("type")
            tag = f"{ft.get('type')}: {d}" if d and d != ft.get("type") else ft.get("type")
            if tag not in sites:
                sites.append(tag)
    return {
        "protein_name": pn,
        "function": (fn_text[:400] if fn_text else None),
        "keywords": kws[:10],
        "sites": sites[:8],
    }


def main() -> None:
    pairs = pd.read_csv(ROOT / "data/proteingym/fitness_top_features_auxfix.csv")
    ref = pd.read_csv(ROOT / "data/external/DMS_substitutions.csv",
                      usecols=["DMS_id", "UniProt_ID"])
    ids = sorted(ref[ref.DMS_id.isin(pairs.DMS_id.unique())]["UniProt_ID"].dropna().unique())
    print(f"{len(ids)} unique assay proteins to fetch")

    by_entryname: dict[str, dict] = {}
    session = requests.Session()
    for b in range(0, len(ids), BATCH):
        chunk = ids[b:b + BATCH]
        q = " OR ".join(f"(id:{x})" for x in chunk)
        params = {"query": q, "fields": FIELDS, "format": "json", "size": 500}
        for attempt in range(4):
            try:
                resp = session.get(SEARCH, params=params, timeout=60)
                resp.raise_for_status()
                for e in resp.json().get("results", []):
                    by_entryname[e.get("uniProtkbId")] = summarize(e)
                break
            except Exception as ex:
                print(f"  ! batch {b // BATCH} attempt {attempt + 1}: {ex}")
                time.sleep(2 * (attempt + 1))
        time.sleep(0.3)

    # Map each requested UniProt_ID -> summary (UniProt_ID IS the entry name here).
    out = {}
    missing = []
    for uid in ids:
        if uid in by_entryname:
            out[uid] = by_entryname[uid]
        else:
            missing.append(uid)
            out[uid] = {"protein_name": None, "function": None, "keywords": [], "sites": []}
    print(f"resolved {len(ids) - len(missing)}/{len(ids)} ; missing: {missing[:15]}")

    dest = ROOT / "data/proteingym/assay_protein_function.json"
    with dest.open("w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
