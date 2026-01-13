# PyMUSAS Translation Utilities

This directory contains tools for evaluating and enhancing the PyMUSAS Danish tagger.

## Scripts

### `eval_europarl.py`

Evaluates the performance of the Danish tagger against English "silver standard" tags using the Europarl parallel corpus.

**Usage:**

```bash
# Evaluate 100 sentences (default)
python eval_europarl.py

# Custom configuration
python eval_europarl.py --num-samples 1000 --top-k 5
```

**Output:**
Results are saved to `resources/output/pymusas_eval/`.

**Requirements:**
- Python environment with dependencies from `requirements.txt`
- `da_core_news_sm` and `en_core_web_sm` spaCy models
- `en_dual_none_contextual` PyMUSAS model
- Lexicon files in `resources/pymusas/da/`

### `wiktionary_lookup.py`

Lookups multi-word expression (MWE) candidates on Wiktionary to validate entries and fetch definitions. Input is typically a cleaned MWE list.

### `wikipedia_lookup.py`

Performs validation and definition fetching for MWEs using Wikipedia data, often used as a fallback or complementary source to Wiktionary.

### `wordnet_lookup.py`

Lookups MWEs in WordNet using NLTK to find synsets and definitions for terms not found in other dictionaries.
