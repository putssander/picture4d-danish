
import csv
import argparse
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Health-related keywords (extend as needed)
KEYWORDS = [
    "health", "doctor", "medicine", "hospital", "patient", "disease", 
    "treatment", "virus", "infection", "clinical", "medical", "cancer", 
    "surgery", "nurse", "pharmacy", "drug", "healthcare", "symptom"
]

def is_health_related(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in KEYWORDS)

def main():
    parser = argparse.ArgumentParser(description="Extract health-related sentences from Europarl in DA, EN, ES, NL")
    parser.add_argument("--output", type=str, default="europarl_health_aligned.csv", help="Output CSV file path")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum aligned samples to collect")
    parser.add_argument("--candidate-pool", type=int, default=5000, help="Size of candidate pool from first pair")
    args = parser.parse_args()

    # Determine output path relative to resources if not absolute
    # If standard filename, save to resources/output/health_corpus/
    if args.output == "europarl_health_aligned.csv":
        script_dir = Path(__file__).resolve().parent
        # script_dir is ".../pic4dclean/scripts"
        # script_dir.parent is ".../pic4dclean"
        output_dir = script_dir.parent / "resources" / "output" / "health_corpus"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / args.output
    else:
        output_path = Path(args.output)
    
    # Store candidates: 
    # {english_text: { 
    #    'da': ..., 'en': ..., 
    #    'es': ..., 'nl': ...,
    #    'da_en_id': ...,  # Row index in da-en dataset
    #    'en_es_id': ...,  # Row index in en-es dataset
    #    'en_nl_id': ...   # Row index in en-nl dataset
    # }}
    candidates = {}
    
    # Phase 1: Danish-English (The Anchor)
    # We collect English sentences that are health-related and their Danish translations
    print(f"--- Phase 1: Scanning da-en for {args.candidate_pool} health candidates ---")
    ds_da = load_dataset("Helsinki-NLP/europarl", "da-en", split="train", streaming=True, trust_remote_code=True)
    
    # We need to track the global index in the stream
    for idx, sample in tqdm(enumerate(ds_da)):
        en_text = sample['translation']['en']
        da_text = sample['translation']['da']
        
        if is_health_related(en_text):
            candidates[en_text] = {
                'da': da_text, 
                'en': en_text,
                'da_en_id': f"europarl:da-en:{idx}"
            }
            if len(candidates) >= args.candidate_pool:
                break
    
    print(f"Collected {len(candidates)} candidates.")

    # Phase 2: English-Spanish
    # We check which of our candidates exist in the Spanish dataset
    print("\n--- Phase 2: Scanning en-es to align Spanish ---")
    ds_es = load_dataset("Helsinki-NLP/europarl", "en-es", split="train", streaming=True, trust_remote_code=True)
    
    needed_en = set(candidates.keys())
    
    for idx, sample in tqdm(enumerate(ds_es)):
        en_text = sample['translation']['en']
        if en_text in needed_en:
            candidates[en_text]['es'] = sample['translation']['es']
            candidates[en_text]['en_es_id'] = f"europarl:en-es:{idx}"
    
    # Filter out candidates that didn't get Spanish
    candidates = {k: v for k, v in candidates.items() if 'es' in v}
    print(f"Candidates remaining after Spanish alignment: {len(candidates)}")
    if len(candidates) < args.min_samples:
         print(f"Warning: Only {len(candidates)} samples left. You might need a larger candidate pool.")

    # Phase 3: English-Dutch
    print("\n--- Phase 3: Scanning en-nl to align Dutch ---")
    ds_nl = load_dataset("Helsinki-NLP/europarl", "en-nl", split="train", streaming=True, trust_remote_code=True)
    
    needed_en = set(candidates.keys())
    
    for idx, sample in tqdm(enumerate(ds_nl)):
        en_text = sample['translation']['en']
        if en_text in needed_en:
            candidates[en_text]['nl'] = sample['translation']['nl']
            candidates[en_text]['en_nl_id'] = f"europarl:en-nl:{idx}"

    # Final Filter
    final_sentences = [v for v in candidates.values() if 'nl' in v]
    print(f"\nFinal count of 4-way aligned sentences: {len(final_sentences)}")
    
    if len(final_sentences) < args.min_samples:
        print("Warning: Did not reach desired minimum samples.")
        
    # Formatting for CSV
    data_rows = []
    for item in final_sentences:
        data_rows.append({
            'en': item['en'],
            'da': item['da'],
            'es': item['es'],
            'nl': item['nl'],
            'da_en_id': item['da_en_id'],
            'en_es_id': item['en_es_id'],
            'en_nl_id': item['en_nl_id']
        })

    # Saving
    print(f"\nSaving results to {output_path}...")
    fieldnames = ['en', 'da', 'es', 'nl', 'da_en_id', 'en_es_id', 'en_nl_id']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
    
    print("Done.")

if __name__ == "__main__":
    main()
