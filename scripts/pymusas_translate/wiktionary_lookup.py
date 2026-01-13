#!/usr/bin/env python3
"""
wiktionary_lookup.py

Batch‑lookup MWEs on Wiktionary with a tight request budget (≤ 2 API calls per
row). Uses the Revisions API to fetch wikitext and extract the first
English definition bullet.

Request budget
--------------
* **0 calls** if both URL and definition already in the row.
* **1 call** if URL present but definition missing.
* **≤ 2 calls** if URL absent – direct Revisions by title guess, then
  fallback search → Revisions.

Output columns (prefixed **wiktionary_**)
----------------------------------------
* **wiktionary_url** – canonical page URL
* **wiktionary_definition** – first English `#` definition plain text
* **wiktionary_sanity_check** – `True` if every word appears in the URL

Other behaviour
---------------
* Default input: `mwe_en_cleaned.xlsx` (override with `--input`).
* Default subset: reproducible 30‑row sample (seed 42). Use `--sample 0` for all.
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
    ap = argparse.ArgumentParser(description="Batch Wiktionary lookup (30‑row sample by default).")
    ap.add_argument("--input", default="mwe_en_cleaned.xlsx", help="Dataframe path (CSV/XLSX/Pickle).")
    ap.add_argument("--sample", type=int, metavar="N", default=30, help="Sample N rows (0 = all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    return ap.parse_args()

# ───────────────────── constants & regexes ──────────────────

USER_AGENT   = "Mozilla/5.0 (compatible; wiktionary-lookup/2.7 +https://github.com/yourname/yourrepo)"
REQUEST_DELAY = 1.0
ASTERISK_RE  = re.compile(r"\*")
WIKI_BASE    = "https://en.wiktionary.org/wiki/"
API          = "https://en.wiktionary.org/w/api.php"

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

def revisions_request(title: str) -> Optional[str]:
    """Fetch raw wikitext for *title* or None."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "content",
        "rvslots": "main",
    }
    try:
        res = requests.get(API, params=params, timeout=10, headers={"User-Agent": USER_AGENT}).json()
        pages = res.get("query", {}).get("pages", {})
        rev = next(iter(pages.values()), {})
        slots = rev.get("revisions", [{}])[0].get("slots", {})
        return slots.get("main", {}).get("*", "")
    except requests.RequestException:
        return None


def parse_first_english_definition(wikitext: str) -> Optional[str]:
    """Extract first `#` bullet in the English section."""
    lines = wikitext.split("\n")
    in_english = False
    for ln in lines:
        if ln.startswith("==") and "English" in ln:
            in_english = True
            continue
        if in_english and ln.startswith("=="):
            break
        if in_english and ln.lstrip().startswith("#"):
            return f"wiktionary: {ln.lstrip('#').strip()}"
    for ln in lines:
        if ln.lstrip().startswith("#"):
            return f"wiktionary: {ln.lstrip('#').strip()}"
    return None


def definition_from_title(title: str) -> Optional[str]:
    wt = revisions_request(title)
    return parse_first_english_definition(wt) if wt else None


def search_request(query: str) -> Optional[str]:
    """Return best-match page title via search API, else None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
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


def derive_url_and_definition(phrase: str) -> Tuple[Optional[str], Optional[str]]:
    base_phrase = unicodedata.normalize("NFKC", strip_wildcards(phrase.strip()))
    guess = base_phrase.replace(" ", "_")

    # direct revision lookup
    definition = definition_from_title(guess)
    if definition:
        return WIKI_BASE + quote(guess), definition

    # fallback: search → revision
    title = search_request(base_phrase)
    if title:
        definition = definition_from_title(title)
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

    work_df = df.sample(args.sample, random_state=args.seed) if args.sample else df.copy()

    for col in ("wiktionary_url", "wiktionary_definition", "wiktionary_sanity_check"):
        if col not in work_df.columns:
            work_df[col] = pd.NA

    urls, defs, checks = [], [], []
    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Wiktionary lookup"):
        phrase = row["pos_cleaned"]
        url = row.get("wiktionary_url") if pd.notna(row.get("wiktionary_url")) else None
        definition = row.get("wiktionary_definition") if pd.notna(row.get("wiktionary_definition")) else None

        if url and not definition:
            title = title_from_url(url)
            if title:
                definition = definition_from_title(title)
        if not url:
            url, definition = derive_url_and_definition(phrase)

        urls.append(url)
        defs.append(definition)
        checks.append(terms_in_url(phrase, url))
        time.sleep(REQUEST_DELAY)

    work_df["wiktionary_url"] = urls
    work_df["wiktionary_definition"] = defs
    work_df["wiktionary_sanity_check"] = checks

    if args.sample:
        out_path = in_path.with_name(f"{in_path.stem}_wiktionary_{args.sample}.csv")
        work_df.to_csv(out_path, index=False)
        print("✅ Subset written to", out_path)
    else:
        print("✅ Lookup complete – full dataframe processed (no CSV written).")

if __name__ == "__main__":
    main()
