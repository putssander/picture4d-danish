#!/usr/bin/env python3
"""
wordnet_lookup.py

Batch‑lookup MWEs in WordNet using NLTK. Designed for entries where both 
Wiktionary and Wikipedia lookups failed or have no sanity check.

Features
--------
* Looks up multi-word expressions in WordNet synsets
* Extracts definitions from synset glosses
* Configurable filtering based on sanity check columns
* Returns synset names and combined definitions

Output columns (prefixed **wordnet_**)
--------------------------------------
* **wordnet_synsets** – comma-separated list of synset IDs
* **wordnet_definition** – combined definitions from all synsets
* **wordnet_sanity_check** – `True` if lemma found in WordNet

Default behavior
---------------
* Only processes rows where both wiktionary_sanity_check and wikipedia_sanity_check are False
* Or rows where these columns are missing/NaN
* Configurable with --filter-mode
* Default input: processes all matching rows
* Strips `*` wildcards anywhere in the phrase
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import unicodedata
from typing import Optional, List, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import nltk
    from nltk.corpus import wordnet as wn
    # Download WordNet if not already present
    try:
        wn.synsets('test')  # Test if WordNet is available
    except LookupError:
        print("📦 Downloading WordNet...")
        nltk.download('wordnet', quiet=True)
except ImportError:
    print("❌ NLTK not installed. Please run: pip install nltk")
    exit(1)

# ────────────────────────── CLI ──────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch WordNet lookup for failed dictionary entries.")
    ap.add_argument("--input", default="mwe_en_cleaned.xlsx", help="Dataframe path (CSV/XLSX/Pickle).")
    ap.add_argument("--sample", type=int, metavar="N", default=0, help="Sample N rows (0 = all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--filter-mode", choices=["both-false", "any-false", "missing", "all"], 
                    default="both-false",
                    help="Filter mode: both-false (both wikt+wiki false), any-false (either false), missing (no sanity checks), all (process all)")
    ap.add_argument("--force-all", action="store_true", 
                    help="Process all rows regardless of sanity check status (same as --filter-mode all).")
    return ap.parse_args()

# ───────────────────── constants & regexes ──────────────────

ASTERISK_RE = re.compile(r"\*")

# ───────────────────── helper functions ──────────────────

def strip_wildcards(text: str) -> str:
    """Remove asterisk wildcards from text."""
    return ASTERISK_RE.sub("", text)

def normalize_for_wordnet(text: str) -> str:
    """Normalize text for WordNet lookup."""
    # Remove wildcards and normalize unicode
    text = unicodedata.normalize("NFKC", strip_wildcards(text.strip()))
    # WordNet uses underscores for multi-word lemmas
    return text.lower().replace(" ", "_").replace("-", "_")

def should_process_row(row: pd.Series, filter_mode: str) -> bool:
    """Determine if a row should be processed based on filter mode."""
    wikt_check = row.get("wiktionary_sanity_check")
    wiki_check = row.get("wikipedia_sanity_check")
    
    if filter_mode == "all":
        return True
    elif filter_mode == "both-false":
        return (wikt_check is False or pd.isna(wikt_check)) and (wiki_check is False or pd.isna(wiki_check))
    elif filter_mode == "any-false":
        return wikt_check is False or wiki_check is False or pd.isna(wikt_check) or pd.isna(wiki_check)
    elif filter_mode == "missing":
        return pd.isna(wikt_check) and pd.isna(wiki_check)
    else:
        return False

# ─────────────── WordNet lookup functions ────────────────

def lookup_wordnet_synsets(phrase: str) -> Tuple[List[str], str, bool]:
    """
    Look up phrase in WordNet and return synsets, definitions, and success flag.
    
    Returns:
        synsets: List of synset names (e.g., ['dog.n.01', 'dog.v.01'])
        definition: Combined definitions from all synsets
        found: True if any synsets were found
    """
    normalized_phrase = normalize_for_wordnet(phrase)
    
    # Try different variations
    variations = [
        normalized_phrase,                    # exact normalization
        normalized_phrase.replace("_", ""),   # compound word
        phrase.lower().replace(" ", "_"),     # preserve original spacing
        phrase.lower().replace("-", "_"),     # hyphen to underscore
    ]
    
    all_synsets = []
    all_definitions = []
    
    for variation in variations:
        try:
            synsets = wn.synsets(variation)
            for synset in synsets:
                synset_name = synset.name()
                if synset_name not in all_synsets:
                    all_synsets.append(synset_name)
                    # Get definition from synset gloss
                    definition = synset.definition()
                    if definition:
                        all_definitions.append(f"{synset_name}: {definition}")
        except Exception:
            continue
    
    # Also try to find as lemma in synsets (for multi-word expressions)
    try:
        for synset in wn.all_synsets():
            for lemma in synset.lemmas():
                lemma_name = lemma.name().lower()
                if lemma_name == normalized_phrase or lemma_name == phrase.lower().replace(" ", "_"):
                    synset_name = synset.name()
                    if synset_name not in all_synsets:
                        all_synsets.append(synset_name)
                        definition = synset.definition()
                        if definition:
                            all_definitions.append(f"{synset_name}: {definition}")
    except Exception:
        pass
    
    combined_definition = " | ".join(all_definitions) if all_definitions else ""
    if combined_definition:
        combined_definition = f"wordnet: {combined_definition}"
    
    return all_synsets, combined_definition, len(all_synsets) > 0

# ──────────────────────────── main ────────────────────────────

def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    
    # Override filter mode if force-all is used
    if args.force_all:
        args.filter_mode = "all"

    # Load dataframe
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

    # Filter rows based on criteria
    filtered_indices = []
    for idx, row in df.iterrows():
        if should_process_row(row, args.filter_mode):
            filtered_indices.append(idx)
    
    if not filtered_indices:
        print(f"⚠️  No rows match filter criteria '{args.filter_mode}'. Try a different --filter-mode.")
        return
    
    filtered_df = df.loc[filtered_indices].copy()
    print(f"📊 Filtering to {len(filtered_df)} rows using filter mode '{args.filter_mode}'")
    
    # Sample if requested
    work_df = filtered_df.sample(args.sample, random_state=args.seed) if args.sample else filtered_df.copy()
    
    # Initialize WordNet columns
    for col in ("wordnet_synsets", "wordnet_definition", "wordnet_sanity_check"):
        if col not in work_df.columns:
            work_df[col] = pd.NA

    # Process each row
    synsets_list, definitions_list, checks_list = [], [], []
    
    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="WordNet lookup"):
        phrase = row["pos_cleaned"]
        
        # Skip if already processed
        if pd.notna(row.get("wordnet_definition")):
            synsets_list.append(row.get("wordnet_synsets"))
            definitions_list.append(row.get("wordnet_definition"))
            checks_list.append(row.get("wordnet_sanity_check"))
            continue
        
        synsets, definition, found = lookup_wordnet_synsets(phrase)
        
        synsets_list.append(", ".join(synsets) if synsets else None)
        definitions_list.append(definition if definition else None)
        checks_list.append(found)

    # Update dataframe
    work_df["wordnet_synsets"] = synsets_list
    work_df["wordnet_definition"] = definitions_list
    work_df["wordnet_sanity_check"] = checks_list

    # Prepare output with unprocessed rows if not processing all
    if args.filter_mode != "all":
        # Get unprocessed rows
        unprocessed_indices = [idx for idx in df.index if idx not in filtered_indices]
        if unprocessed_indices:
            unprocessed_df = df.loc[unprocessed_indices].copy()
            
            # Add empty WordNet columns to unprocessed rows
            for col in ("wordnet_synsets", "wordnet_definition", "wordnet_sanity_check"):
                if col not in unprocessed_df.columns:
                    unprocessed_df[col] = pd.NA
            
            # Combine processed and unprocessed rows
            output_df = pd.concat([work_df, unprocessed_df], ignore_index=False)
            output_df = output_df.sort_index()  # Maintain original order
        else:
            output_df = work_df
    else:
        output_df = work_df

    # Generate output filename
    if args.sample:
        suffix = f"_wordnet_{args.sample}_{args.filter_mode}"
    else:
        suffix = f"_wordnet_all_{args.filter_mode}"
    
    out_path = in_path.with_name(f"{in_path.stem}{suffix}.csv")
    output_df.to_csv(out_path, index=False)
    
    # Print statistics
    success_count = sum(1 for check in checks_list if check)
    print("✅ Results written to", out_path)
    print(f"📊 Processed {len(work_df)} rows, found {success_count} WordNet matches ({(success_count/len(work_df)*100):.1f}%)")
    print(f"📊 Output contains {len(work_df)} processed rows and {len(output_df) - len(work_df)} unprocessed rows")

if __name__ == "__main__":
    main()