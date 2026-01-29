#!/usr/bin/env python3
"""
download_mwe.py

Download the UCREL English MWE list (English), save the raw data, and add a
**pos_cleaned** column that applies the *minimal* normalisation requested:

1. **Split on whitespace** to treat each token separately.
2. For every token
   * If the token is a gap specification like `{ADJ/INTJ}` – **replace it with
     a single `*`**.
   * Otherwise strip any POS suffix that starts with an underscore, including
     the wildcard form `_*` (e.g. `Antique_*` → `Antique`).
3. Re-join the tokens with single spaces and collapse any accidental double
   spaces.

Standalone `*` tokens that already occur in the template stay exactly as they
are. No other characters are changed. Duplicates based on **pos_cleaned** are
reported only (never removed). Both raw and cleaned Excel files are written to
disk.
"""

from pathlib import Path
import re
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────
# ── Constants ─────────────────────────────────────────────────────
RAW_URL = (
    "https://raw.githubusercontent.com/UCREL/Multilingual-USAS/refs/heads/master/English/"
    "mwe-en.tsv"
)

# Base directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent

# English resources directory: ../../resources/pymusas/en
EN_DIR = SCRIPT_DIR.parents[1] / "resources" / "pymusas" / "en"
EN_DIR.mkdir(parents=True, exist_ok=True)

RAW_XLSX   = EN_DIR / "mwe_en.xlsx"
CLEAN_XLSX = EN_DIR / "mwe_en_cleaned.xlsx"

# ── Helper functions ──────────────────────────────────────────────

def load_raw_mwes(url: str) -> pd.DataFrame:
    """Download the TSV file located at *url* into a DataFrame."""
    return pd.read_csv(url, delimiter="\t", header=0)


def save_excel(df: pd.DataFrame, path: Path, label: str) -> None:
    """Save *df* to *path* (Excel) and log a message."""
    df.to_excel(path, index=False)
    print(f"{label} data ➜ {path.resolve()}")


# Cleaning ---------------------------------------------------------
POS_SUFFIX_RE   = re.compile(r"_(?:[A-Z]+|\*)$")   # trailing _NOUN, _PROPN or _*
GAP_TOKEN_RE    = re.compile(r"^\{[^}]*\}$")       # token that is exactly {...}
WHITESPACE_RE   = re.compile(r"\s+")


def clean_mwe(text: str) -> str:
    """Return cleaned MWE template string according to the three-step rules."""
    if not isinstance(text, str):
        text = str(text)

    tokens_in  = WHITESPACE_RE.split(text.strip())
    tokens_out = []

    for tok in tokens_in:
        # 1) gap block → '*'
        if GAP_TOKEN_RE.match(tok):
            tokens_out.append("*")
            continue
        # 2) remove trailing POS or wildcard suffix
        tok = POS_SUFFIX_RE.sub("", tok)
        tokens_out.append(tok)

    # 3) join and normalise whitespace
    cleaned = " ".join(tok for tok in tokens_out if tok).strip()
    return cleaned


def add_clean_column(df: pd.DataFrame) -> None:
    """Add **pos_cleaned** column using `clean_mwe`."""
    if "mwe_template" not in df.columns:
        raise KeyError("'mwe_template' column not found in the input TSV!")
    df["pos_cleaned"] = df["mwe_template"].apply(clean_mwe)


def report_duplicates(df: pd.DataFrame) -> None:
    """Print how many duplicates exist based on **pos_cleaned** (no rows dropped)."""
    dups = df[df.duplicated("pos_cleaned", keep=False)]
    print(f"Duplicate cleaned MWEs: {len(dups):,}")
    if not dups.empty:
        print(dups[["mwe_template", "pos_cleaned"]].head())


# ── Main routine ──────────────────────────────────────────────────

def main() -> None:
    df = load_raw_mwes(RAW_URL)
    print(f"Total entries (raw): {len(df):,}")
    save_excel(df, RAW_XLSX, "Raw")

    add_clean_column(df)
    report_duplicates(df)

    save_excel(df, CLEAN_XLSX, "Cleaned")


if __name__ == "__main__":
    main()
