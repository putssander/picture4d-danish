# 4dpicture-Danish

This repository represents the output of the Danish team for the [4DPICTURE](https://4dpicture.eu) project. The primary objective is to create a Danish version of [The Metaphor Menu](https://wp.lancs.ac.uk/melc/the-metaphor-menu/). To facilitate the extraction of metaphors for this purpose, we utilize PyMUSAS. As part of this effort, we have developed and evaluated a Danish version of the PyMUSAS tool.

## Scripts

- **[PyMUSAS Translation Utilities](scripts/pymusas_translate/README.md)**: Scripts for evaluating tagging performance (`eval_europarl.py`) and performing dictionary lookups (`wiktionary_lookup.py`, etc.).

## Setup

### Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

1.  Create a virtual environment:
    ```bash
    python3 -m venv .venv
    ```

2.  Activate the virtual environment:
    -   On macOS/Linux: `source .venv/bin/activate`
    -   On Windows: `.venv\Scripts\activate`

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Prerequisites

1.  **Python Packages**: Ensure you have the necessary dependencies installed (using the `requirements.txt` file as shown above).

2.  **Spacy Models**: Download the required language models:
    ```bash
    python -m spacy download da_core_news_sm
    python -m spacy download en_core_web_sm
    ```
    
    The script also requires the `en_dual_none_contextual` PyMUSAS model. This is included in `requirements.txt`, but if you need to install it manually:
    ```bash
    pip install https://github.com/UCREL/pymusas-models/releases/download/en_dual_none_contextual-0.3.1/en_dual_none_contextual-0.3.1-py3-none-any.whl
    ```

3.  **Lexicon Files**: The script expects the lexicon files to be located in the project's resources directory:
    - `resources/pymusas/da/semantic_lexicon_da_clean.tsv`
    - `resources/pymusas/da/mwe_da_clean.tsv`
