#!/usr/bin/env python3
"""
wikipedia_lookup.py

Batch‑lookup MWEs on Wikipedia with a tight request budget (≤ 2 API calls per
row). Uses the Extracts API to fetch clean definitions and summaries.

Request budget
--------------
* **0 calls** if both URL and definition already in the row.
* **1 call** if URL present but definition missing.
* **≤ 2 calls** if URL absent – direct page lookup by title guess, then
  fallback search → page lookup.

Output columns (prefixed **wikipedia_**)
----------------------------------------
* **wikipedia_url** – canonical page URL
* **wikipedia_definition** – page extract/summary
* **wikipedia_sanity_check** – `True` if every word appears in the URL

Default behavior
---------------
* Only processes rows where `wiktionary_sanity_check` is False (changeable with --force-all).
* Default input: `mwe_en_cleaned.xlsx` (override with `--input`).
* Default subset: reproducible 30‑row sample (seed 42). Use `--sample 0` for all.
* Strips `*` wildcards anywhere in the phrase.
* Writes only processed rows when sampling is active.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time
import unicodedata
from typing import Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import quote

# ────────────────────────── CLI ──────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch Wikipedia lookup (30‑row sample by default).")
    ap.add_argument("--input", default="mwe_en_cleaned.xlsx", help="Dataframe path (CSV/XLSX/Pickle).")
    ap.add_argument("--sample", type=int, metavar="N", default=30, help="Sample N rows (0 = all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--force-all", action="store_true", 
                    help="Process all rows regardless of wiktionary_sanity_check status.")
    ap.add_argument("--extract-length", type=int, default=300, 
                    help="Maximum extract length in characters (default: 300).")
    return ap.parse_args()

# ───────────────────── constants & regexes ──────────────────

USER_AGENT   = "Mozilla/5.0 (compatible; wikipedia-lookup/1.0 +https://github.com/yourname/yourrepo)"
REQUEST_DELAY = 1.0
ASTERISK_RE  = re.compile(r"\*")
WIKI_BASE    = "https://en.wikipedia.org/wiki/"
API          = "https://en.wikipedia.org/w/api.php"

# ───────────────────── helper functions ──────────────────

def strip_wildcards(text: str) -> str:
    return ASTERISK_RE.sub("", text)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", strip_wildcards(text).lower())


def terms_in_url(mwe: str, url: str | None) -> bool:
    if not url:
        return False
    norm_url = normalize(url)
    return all(normalize(tok) in norm_url for tok in mwe.split() if tok)


def title_from_url(url: str) -> Optional[str]:
    return url.split("/wiki/")[1].split("#")[0] if url and url.startswith(WIKI_BASE) else None

# ─────────────── API request helpers ────────────────

def extracts_request(title: str, extract_length: int = 300) -> Optional[str]:
    """Fetch page extract for *title* or None."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "exintro": True,
        "explaintext": True,
        "exchars": extract_length,
        "redirects": True,  # Follow redirects to canonical pages
        "uselang": "en",    # Force English language interface
    }
    try:
        res = requests.get(API, params=params, timeout=10, headers={"User-Agent": USER_AGENT}).json()
        pages = res.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        
        # Check if page exists (no "missing" key)
        if "missing" in page:
            return None
            
        extract = page.get("extract", "").strip()
        return f"wikipedia: {extract}" if extract else None
    except requests.RequestException:
        return None


def definition_from_title(title: str, extract_length: int = 300) -> Optional[str]:
    return extracts_request(title, extract_length)


def search_request(query: str) -> Optional[str]:
    """Return best-match page title via search API, else None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
        "srnamespace": 0,  # Main namespace only
        "uselang": "en",   # Force English language interface
        "srwhat": "text",  # Search in page text
    }
    try:
        res = requests.get(API, params=params, timeout=10, headers={"User-Agent": USER_AGENT}).json()
        hits = res.get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0].get("title")
        return title.replace(" ", "_") if title else None
    except requests.RequestException:
        return None


def derive_url_and_definition(phrase: str, extract_length: int = 300) -> Tuple[Optional[str], Optional[str]]:
    base_phrase = unicodedata.normalize("NFKC", strip_wildcards(phrase.strip()))
    guess = base_phrase.replace(" ", "_")

    # direct page lookup
    definition = definition_from_title(guess, extract_length)
    if definition:
        return WIKI_BASE + quote(guess), definition

    # fallback: search → page lookup
    title = search_request(base_phrase)
    if title:
        definition = definition_from_title(title, extract_length)
        if definition:
            return WIKI_BASE + quote(title), definition
    return None, None

# ──────────────────────────── main ────────────────────────────

def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    if in_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(in_path, engine="openpyxl")
    elif in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.suffix.lower() == ".tsv":
        df = pd.read_csv(in_path, sep='\t')
    else:
        df = pd.read_pickle(in_path)

    if "pos_cleaned" not in df.columns:
        raise KeyError("'pos_cleaned' column not found.")

    # Filter based on wiktionary_sanity_check unless --force-all is used
    if not args.force_all and "wiktionary_sanity_check" in df.columns:
        # Only process rows where wiktionary_sanity_check is False
        filtered_df = df[df["wiktionary_sanity_check"] == False].copy()
        if len(filtered_df) == 0:
            print("⚠️  No rows with wiktionary_sanity_check=False found. Use --force-all to process all rows.")
            return
        print(f"📊 Filtering to {len(filtered_df)} rows where wiktionary_sanity_check=False")
    else:
        filtered_df = df.copy()
        if not args.force_all:
            print("⚠️  No wiktionary_sanity_check column found. Processing all rows.")

    work_df = filtered_df.sample(args.sample, random_state=args.seed) if args.sample else filtered_df.copy()

    for col in ("wikipedia_url", "wikipedia_definition", "wikipedia_sanity_check"):
        if col not in work_df.columns:
            work_df[col] = pd.NA

    urls, defs, checks = [], [], []
    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Wikipedia lookup"):
        phrase = row["pos_cleaned"]
        url = row.get("wikipedia_url") if pd.notna(row.get("wikipedia_url")) else None
        definition = row.get("wikipedia_definition") if pd.notna(row.get("wikipedia_definition")) else None

        if url and not definition:
            title = title_from_url(url)
            if title:
                definition = definition_from_title(title, args.extract_length)
        if not url:
            url, definition = derive_url_and_definition(phrase, args.extract_length)

        urls.append(url)
        defs.append(definition)
        checks.append(terms_in_url(phrase, url))
        time.sleep(REQUEST_DELAY)

    work_df["wikipedia_url"] = urls
    work_df["wikipedia_definition"] = defs
    work_df["wikipedia_sanity_check"] = checks

    # When filtering is applied, also include the rows with wiktionary_sanity_check=True for context
    if not args.force_all and "wiktionary_sanity_check" in df.columns:
        # Get the rows that were NOT processed (wiktionary_sanity_check=True)
        unprocessed_df = df[df["wiktionary_sanity_check"] == True].copy()
        
        # Add empty wikipedia columns to unprocessed rows
        for col in ("wikipedia_url", "wikipedia_definition", "wikipedia_sanity_check"):
            if col not in unprocessed_df.columns:
                unprocessed_df[col] = pd.NA
        
        # Combine processed and unprocessed rows
        output_df = pd.concat([work_df, unprocessed_df], ignore_index=True)
        
        # Sort by original index to maintain order
        if 'index' in output_df.columns:
            output_df = output_df.sort_values('index').reset_index(drop=True)
    else:
        output_df = work_df

    # Always write output file
    if args.sample:
        suffix = f"_wikipedia_{args.sample}"
        if not args.force_all:
            suffix += "_filtered"
    else:
        suffix = "_wikipedia_all"
        if not args.force_all:
            suffix += "_filtered"
    
    out_path = in_path.with_name(f"{in_path.stem}{suffix}.csv")
    output_df.to_csv(out_path, index=False)
    print("✅ Results written to", out_path)
    print(f"📊 Output contains {len(work_df)} processed rows and {len(output_df) - len(work_df)} unprocessed rows")

if __name__ == "__main__":
    main()