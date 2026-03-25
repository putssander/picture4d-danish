# 4dpicture-Danish

This repository represents the output of the Danish team for the [4DPICTURE](https://4dpicture.eu) project. The primary objective is to create a Danish version of [The Metaphor Menu](https://wp.lancs.ac.uk/melc/the-metaphor-menu/). To facilitate the extraction of metaphors for this purpose, we utilize PyMUSAS. As part of this effort, we have developed and evaluated a Danish version of the PyMUSAS tool.

In addition, we performed speech-to-text transcription on interview data from cancer patients, which is used to identify and analyze metaphor usage.




## Speech-to-Text (STT) Transcription of Interviews

To support metaphor analysis, we converted recorded interviews with cancer patients into written text using speech-to-text (STT) technology. This makes it possible to systematically search for and analyze metaphor usage in spoken language.

For transparency and reproducibility, the full workflow is available as notebooks:

- Convert audio files to the correct format (WAV):  
  `notebooks/stt/convert_m4a_to_wav.ipynb`

- Perform fast and efficient transcription using Whisper:  
  `notebooks/stt/transcribe-insanely-fast-whisper.ipynb`

We use a maintained fork of the Whisper-based transcription tool to ensure compatibility and stability:

- [insanely-fast-whisper (maintained fork)](https://github.com/putssander/insanely-fast-whisper)

This fork was created because the original project had unresolved dependency issues, and the updates ensure the transcription pipeline continues to work reliably.


## PyMUSAS translation

Efforts
1. Use google translate for single term translations
2. GPT4o for multi word expressions
3. Opensource definitions
4. GPT internet access


## Scripts

- **[PyMUSAS Translation Utilities](scripts/pymusas_translate/README.md)**: Scripts for evaluating tagging performance (`eval_europarl.py`) and performing dictionary lookups (`wiktionary_lookup.py`, etc.).
- **Health Corpus Creator**: `scripts/create_health_corpus.py` extracts aligned health-related sentences from Europarl.
  ```bash
  python scripts/create_health_corpus.py --min-samples 100 --candidate-pool 10000
  ```

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
