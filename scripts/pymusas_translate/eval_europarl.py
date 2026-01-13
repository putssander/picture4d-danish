# https://huggingface.co/datasets/Helsinki-NLP/europarl/viewer/da-en?views%5B%5D=da_en

"""
Evaluate Danish PyMUSAS tagger against English PyMUSAS tagger using Europarl parallel corpus.
English tags serve as silver standard for evaluating Danish tagger performance.

Usage:
    # Default: 100 sentences, top-1
    python eval_europarl.py
    
    # Custom configuration
    python eval_europarl.py --num-samples 1000 --top-k 5
"""

import argparse
import spacy
import pandas as pd
from datasets import load_dataset
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from pymusas.rankers.lexicon_entry import ContextualRuleBasedRanker
from pymusas.lexicon_collection import LexiconCollection, MWELexiconCollection
from pymusas.taggers.rules.single_word import SingleWordRule
from pymusas.taggers.rules.mwe import MWERule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Danish PyMUSAS tagger against English using Europarl corpus"
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=100,
        help='Number of parallel sentences to evaluate (default: 100)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        choices=[1, 5],
        help='Consider top-k predictions: 1 or 5 (default: 1)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize tags by removing +/- and subcategories (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        dest='normalize',
        action='store_false',
        help='Do not normalize tags'
    )
    return parser.parse_args()


def setup_danish_tagger():
    """Setup Danish PyMUSAS tagger with custom lexicons."""
    nlp = spacy.load("da_core_news_sm", disable=['ner'])
    
    # Locate resources relative to this script
    # Script is in scripts/pymusas_translate/
    # Resources are in resources/pymusas/da/
    script_dir = Path(__file__).resolve().parent
    lexicon_dir = script_dir.parents[1] / 'resources' / 'pymusas' / 'da'
    
    # Single word lexicon
    single_lexicon_url = str(lexicon_dir / 'semantic_lexicon_da_clean.tsv')
    single_lexicon = LexiconCollection.from_tsv(single_lexicon_url)
    single_lemma_lexicon = LexiconCollection.from_tsv(single_lexicon_url, include_pos=False)
    single_rule = SingleWordRule(single_lexicon, single_lemma_lexicon)
    
    # MWE lexicon
    mwe_lexicon_url = str(lexicon_dir / 'mwe_da_clean.tsv')
    mwe_lexicon = MWELexiconCollection.from_tsv(mwe_lexicon_url)
    mwe_rule = MWERule(mwe_lexicon)
    
    rules = [single_rule, mwe_rule]
    ranker = ContextualRuleBasedRanker(*ContextualRuleBasedRanker.get_construction_arguments(rules))
    
    tagger = nlp.add_pipe('pymusas_rule_based_tagger')
    tagger.rules = rules
    tagger.ranker = ranker
    
    return nlp



import en_dual_none_contextual

def setup_english_tagger():
    """Setup English PyMUSAS tagger with official lexicons."""
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    # Load directly to avoid TypeError with newer spacy versions passing 'enable' arg
    english_tagger_pipeline = en_dual_none_contextual.load()
    nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
    
    return nlp


def extract_tags(doc) -> List[List[str]]:
    """Extract USAS tags from spaCy doc."""
    return [token._.pymusas_tags for token in doc]


def normalize_tag(tag: str) -> str:
    """
    Normalize USAS tag by removing polarity markers (+/-) and subcategory markers.
    Example: 'A5.1++' -> 'A5.1', 'B2-' -> 'B2'
    """
    # Remove + and - characters
    tag = tag.rstrip('+-')
    return tag


def tags_match(danish_tags: List[str], english_tags: List[str], 
               top_k: int = 1, normalize: bool = True) -> bool:
    """
    Check if Danish tags match English tags within top-k predictions.
    
    Args:
        danish_tags: List of USAS tags from Danish tagger
        english_tags: List of USAS tags from English tagger (silver standard)
        top_k: Consider top-k predictions
        normalize: Whether to normalize tags (remove +/- and subcategories)
    
    Returns:
        True if any of the top-k Danish tags match any English tag
    """
    if not danish_tags or not english_tags:
        return False
    
    danish_top_k = danish_tags[:top_k]
    
    if normalize:
        danish_top_k = [normalize_tag(tag) for tag in danish_top_k]
        english_tags = [normalize_tag(tag) for tag in english_tags]
    
    return any(dt in english_tags for dt in danish_top_k)


def calculate_sentence_similarity(danish_tags: List[List[str]], 
                                  english_tags: List[List[str]],
                                  top_k: int = 1,
                                  normalize: bool = True) -> float:
    """
    Calculate semantic similarity between Danish and English tag distributions for a sentence.
    Uses Jaccard similarity on tag sets.
    
    Args:
        danish_tags: List of tag lists for each Danish token
        english_tags: List of tag lists for each English token
        top_k: Consider top-k predictions
        normalize: Whether to normalize tags
    
    Returns:
        Jaccard similarity score (0-1)
    """
    # Extract top-k tags from each token and flatten into sets
    danish_set = set()
    for token_tags in danish_tags:
        if token_tags:
            top_tags = token_tags[:top_k]
            if normalize:
                top_tags = [normalize_tag(t) for t in top_tags]
            danish_set.update(top_tags)
    
    english_set = set()
    for token_tags in english_tags:
        if token_tags:
            all_tags = token_tags  # Consider all English tags
            if normalize:
                all_tags = [normalize_tag(t) for t in all_tags]
            english_set.update(all_tags)
    
    # Jaccard similarity: intersection / union
    if not danish_set or not english_set:
        return 0.0
    
    intersection = len(danish_set & english_set)
    union = len(danish_set | english_set)
    
    return intersection / union if union > 0 else 0.0


def calculate_sentence_metrics(danish_sentences: List[List[List[str]]], 
                               english_sentences: List[List[List[str]]],
                               top_k: int = 1,
                               normalize: bool = True) -> Dict[str, float]:
    """
    Calculate evaluation metrics comparing Danish to English tags at sentence level.
    
    Args:
        danish_sentences: List of sentences, each containing list of token tags
        english_sentences: List of sentences, each containing list of token tags
        top_k: Consider top-k predictions
        normalize: Whether to normalize tags
    
    Returns:
        Dictionary with sentence-level similarity metrics
    """
    assert len(danish_sentences) == len(english_sentences), "Must have same number of sentences"
    
    similarities = []
    danish_tag_counts = defaultdict(int)
    english_tag_counts = defaultdict(int)
    common_tags = defaultdict(int)
    danish_only_tags = defaultdict(int)
    english_only_tags = defaultdict(int)
    
    for da_sent, en_sent in zip(danish_sentences, english_sentences):
        # Calculate sentence similarity
        sim = calculate_sentence_similarity(da_sent, en_sent, top_k, normalize)
        similarities.append(sim)
        
        # Collect tag distributions
        da_set = set()
        for token_tags in da_sent:
            if token_tags:
                top_tags = token_tags[:top_k]
                if normalize:
                    top_tags = [normalize_tag(t) for t in top_tags]
                da_set.update(top_tags)
        
        en_set = set()
        for token_tags in en_sent:
            if token_tags:
                all_tags = token_tags
                if normalize:
                    all_tags = [normalize_tag(t) for t in all_tags]
                en_set.update(all_tags)
        
        # Count tag occurrences
        for tag in da_set:
            danish_tag_counts[tag] += 1
        for tag in en_set:
            english_tag_counts[tag] += 1
        
        # Track common and unique tags
        common = da_set & en_set
        da_only = da_set - en_set
        en_only = en_set - da_set
        
        for tag in common:
            common_tags[tag] += 1
        for tag in da_only:
            danish_only_tags[tag] += 1
        for tag in en_only:
            english_only_tags[tag] += 1
    
    # Calculate overall metrics
    mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    # Calculate tag-level precision and recall
    # Precision: Of tags predicted by Danish, how many appear in English?
    # Recall: Of tags in English, how many were predicted by Danish?
    all_danish_tags = set(danish_tag_counts.keys())
    all_english_tags = set(english_tag_counts.keys())
    common_tag_set = all_danish_tags & all_english_tags
    
    precision = len(common_tag_set) / len(all_danish_tags) if all_danish_tags else 0.0
    recall = len(common_tag_set) / len(all_english_tags) if all_english_tags else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'total_sentences': len(danish_sentences),
        'mean_jaccard_similarity': mean_similarity,
        'median_jaccard_similarity': sorted(similarities)[len(similarities)//2] if similarities else 0.0,
        'min_similarity': min(similarities) if similarities else 0.0,
        'max_similarity': max(similarities) if similarities else 0.0,
        'tag_precision': precision,
        'tag_recall': recall,
        'tag_f1': f1,
        'danish_unique_tags': len(all_danish_tags),
        'english_unique_tags': len(all_english_tags),
        'common_unique_tags': len(common_tag_set),
        'similarities': similarities,  # Individual sentence similarities
        'common_tags': dict(sorted(common_tags.items(), key=lambda x: x[1], reverse=True)),
        'danish_only_tags': dict(sorted(danish_only_tags.items(), key=lambda x: x[1], reverse=True)),
        'english_only_tags': dict(sorted(english_only_tags.items(), key=lambda x: x[1], reverse=True))
    }
    
    return metrics


def load_europarl_sample(language_pair: str = "da-en", 
                         split: str = "train", 
                         num_samples: int = 100) -> pd.DataFrame:
    """
    Load sample from Europarl dataset.
    
    Args:
        language_pair: Language pair code (default: "da-en")
        split: Dataset split (train/validation/test)
        num_samples: Number of parallel sentences to load
    
    Returns:
        DataFrame with 'danish' and 'english' columns
    """
    print(f"📦 Loading {num_samples} samples from Europarl {language_pair} dataset...")
    dataset = load_dataset("Helsinki-NLP/europarl", language_pair, split=split)
    
    # Sample the dataset
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Extract parallel sentences
    df = pd.DataFrame({
        'danish': [item['translation']['da'] for item in dataset],
        'english': [item['translation']['en'] for item in dataset]
    })
    
    return df


def evaluate_taggers(danish_nlp, english_nlp, 
                    parallel_corpus: pd.DataFrame,
                    top_k: int = 1,
                    normalize: bool = True,
                    verbose: bool = True) -> Dict:
    """
    Evaluate Danish tagger against English tagger on parallel corpus.
    Uses sentence-level semantic similarity instead of token-level alignment.
    
    Args:
        danish_nlp: Danish spaCy pipeline with PyMUSAS
        english_nlp: English spaCy pipeline with PyMUSAS
        parallel_corpus: DataFrame with 'danish' and 'english' columns
        top_k: Consider top-k predictions
        normalize: Whether to normalize tags
        verbose: Print progress
    
    Returns:
        Dictionary with evaluation results
    """
    danish_sentences = []
    english_sentences = []
    
    print(f"🏷️  Tagging {len(parallel_corpus)} parallel sentences...")
    
    for idx, row in tqdm(parallel_corpus.iterrows(), 
                         total=len(parallel_corpus),
                         disable=not verbose):
        # Process Danish sentence
        da_doc = danish_nlp(row['danish'])
        da_tags = extract_tags(da_doc)  # List[List[str]] - one list per token
        danish_sentences.append(da_tags)
        
        # Process English sentence
        en_doc = english_nlp(row['english'])
        en_tags = extract_tags(en_doc)  # List[List[str]] - one list per token
        english_sentences.append(en_tags)
    
    # Calculate sentence-level metrics
    print("📊 Calculating sentence-level semantic similarity metrics...")
    metrics = calculate_sentence_metrics(danish_sentences, english_sentences, 
                                        top_k=top_k, normalize=normalize)
    
    return metrics


def print_evaluation_report(metrics: Dict, top_k: int):
    """Print formatted evaluation report for sentence-level metrics."""
    print("\n" + "="*70)
    print(f"PyMUSAS Danish vs English Evaluation Report (Top-{top_k})")
    print("Sentence-Level Semantic Similarity")
    print("="*70)
    
    print(f"\n📝 Corpus Statistics:")
    print(f"   Total sentences evaluated: {metrics['total_sentences']:,}")
    print(f"   Danish unique tags: {metrics['danish_unique_tags']:,}")
    print(f"   English unique tags: {metrics['english_unique_tags']:,}")
    print(f"   Common tags (intersection): {metrics['common_unique_tags']:,}")
    
    print(f"\n🎯 Semantic Similarity Scores (Jaccard):")
    print(f"   Mean:   {metrics['mean_jaccard_similarity']:.2%}")
    print(f"   Median: {metrics['median_jaccard_similarity']:.2%}")
    print(f"   Min:    {metrics['min_similarity']:.2%}")
    print(f"   Max:    {metrics['max_similarity']:.2%}")
    
    print(f"\n📊 Tag-Level Metrics:")
    print(f"   Precision: {metrics['tag_precision']:.2%} (Danish tags found in English)")
    print(f"   Recall:    {metrics['tag_recall']:.2%} (English tags found in Danish)")
    print(f"   F1-Score:  {metrics['tag_f1']:.2%}")
    
    print("\n✅ Top 10 Most Common Shared Tags:")
    common_tags = list(metrics['common_tags'].items())[:10]
    for tag, count in common_tags:
        print(f"   {tag}: appeared in {count:,} sentences")
    
    print("\n🇩🇰 Top 10 Danish-Only Tags (not in English):")
    danish_only = list(metrics['danish_only_tags'].items())[:10]
    if danish_only:
        for tag, count in danish_only:
            print(f"   {tag}: appeared in {count:,} sentences")
    else:
        print("   (None - all Danish tags appear in English)")
    
    print("\n🇬🇧 Top 10 English-Only Tags (not in Danish):")
    english_only = list(metrics['english_only_tags'].items())[:10]
    if english_only:
        for tag, count in english_only:
            print(f"   {tag}: appeared in {count:,} sentences")
    else:
        print("   (None - all English tags appear in Danish)")
    
    # Distribution of similarities
    similarities = metrics['similarities']
    high_sim = sum(1 for s in similarities if s >= 0.7)
    medium_sim = sum(1 for s in similarities if 0.4 <= s < 0.7)
    low_sim = sum(1 for s in similarities if s < 0.4)
    
    print(f"\n📈 Similarity Distribution:")
    print(f"   High (≥70%):     {high_sim:,} sentences ({high_sim/len(similarities):.1%})")
    print(f"   Medium (40-70%): {medium_sim:,} sentences ({medium_sim/len(similarities):.1%})")
    print(f"   Low (<40%):      {low_sim:,} sentences ({low_sim/len(similarities):.1%})")
    
    print("\n" + "="*70)


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    print(f"⚙️  Configuration: {args.num_samples} sentences, Top-{args.top_k}, Normalize={args.normalize}")
    print("\n🔧 Setting up taggers...")
    danish_nlp = setup_danish_tagger()
    english_nlp = setup_english_tagger()
    
    print("\n📥 Loading Europarl parallel corpus...")
    parallel_corpus = load_europarl_sample(num_samples=args.num_samples)
    
    print(f"\n📝 Example parallel sentence:")
    print(f"Danish:  {parallel_corpus.iloc[0]['danish'][:100]}...")
    print(f"English: {parallel_corpus.iloc[0]['english'][:100]}...")
    
    # Run evaluation
    metrics = evaluate_taggers(
        danish_nlp, 
        english_nlp, 
        parallel_corpus,
        top_k=args.top_k,
        normalize=args.normalize
    )
    
    # Print report
    print_evaluation_report(metrics, args.top_k)
    
    # Define output directory
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parents[1] / 'resources' / 'output' / 'pymusas_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_data = {
        'total_sentences': [metrics['total_sentences']],
        'mean_jaccard_similarity': [metrics['mean_jaccard_similarity']],
        'median_jaccard_similarity': [metrics['median_jaccard_similarity']],
        'tag_precision': [metrics['tag_precision']],
        'tag_recall': [metrics['tag_recall']],
        'tag_f1': [metrics['tag_f1']],
        'danish_unique_tags': [metrics['danish_unique_tags']],
        'english_unique_tags': [metrics['english_unique_tags']],
        'common_unique_tags': [metrics['common_unique_tags']]
    }
    results_df = pd.DataFrame(results_data)
    output_file = output_dir / f'europarl_evaluation_top{args.top_k}_n{args.num_samples}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Summary results saved to '{output_file}'")
    
    # Save per-sentence similarities
    similarities_df = pd.DataFrame({
        'sentence_id': range(len(metrics['similarities'])),
        'jaccard_similarity': metrics['similarities']
    })
    similarities_file = output_dir / f'europarl_similarities_top{args.top_k}_n{args.num_samples}.csv'
    similarities_df.to_csv(similarities_file, index=False)
    print(f"💾 Per-sentence similarities saved to '{similarities_file}'")


if __name__ == "__main__":
    main()

