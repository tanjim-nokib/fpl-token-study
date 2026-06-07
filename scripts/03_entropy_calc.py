# ============================================================
# 03_entropy_calc.py
# Beyond Syntax — Token Entropy Calculator
# Author: Tanjim Khan Nokib, University of Graz
# Uses character and word entropy from output text
# (Groq does not support logprobs — this is the valid alternative)
# ============================================================

import json, math, os
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR     = 'results/raw'
OUTPUT_PATH = 'results/metrics/entropy.csv'

def compute_char_entropy(text: str) -> float:
    """Shannon entropy over character distribution. H = -Σ p(c)log2(p(c))"""
    if not text:
        return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def compute_word_entropy(text: str) -> float:
    """Shannon entropy over word distribution."""
    words = text.split()
    if not words:
        return 0.0
    counts = Counter(words)
    total  = len(words)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def compute_mean_entropy_from_steps(token_steps: list) -> float:
    """Use pre-computed char entropy if available, else compute from tokens."""
    if not token_steps:
        return 0.0
    # Reconstruct text from token steps
    text = ' '.join(s['token'] for s in token_steps)
    return compute_char_entropy(text)

def process_all_prompts():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])
    if not files:
        print(f'✗ No JSON files in {RAW_DIR}. Run 02_api_collector.py first.')
        return pd.DataFrame()

    records = []
    for fname in tqdm(files, desc='Computing entropy'):
        with open(os.path.join(RAW_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)

        pid  = data['prompt_id']
        meta = data['metadata']

        char_entropies = []
        word_entropies = []

        for run in data['runs']:
            text = run['generated_text']
            # Use pre-computed entropy if available
            if 'char_entropy' in run:
                char_entropies.append(run['char_entropy'])
                word_entropies.append(run['word_entropy'])
            else:
                char_entropies.append(compute_char_entropy(text))
                word_entropies.append(compute_word_entropy(text))

        records.append({
            'prompt_id':        pid,
            'task_name':        meta['task_name'],
            'strategy':         meta['strategy'],
            'length':           meta['length'],
            'specificity':      meta['specificity'],
            'vocabulary':       meta['vocabulary'],
            'syntax':           meta['syntax'],
            'entropy_mean':     float(np.mean(char_entropies)),
            'entropy_std':      float(np.std(char_entropies)),
            'word_entropy_mean':float(np.mean(word_entropies)),
            'word_entropy_std': float(np.std(word_entropies)),
        })

    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Entropy saved to {OUTPUT_PATH}')
    print(f'  Processed: {len(df)} prompts')
    print(f'  Mean char entropy: {df["entropy_mean"].mean():.4f} bits')
    print(f'  Mean word entropy: {df["word_entropy_mean"].mean():.4f} bits')
    return df

if __name__ == '__main__':
    df = process_all_prompts()
    if not df.empty:
        print('\nEntropy by strategy:')
        print(df.groupby('strategy')['entropy_mean'].agg(['mean','std']).round(4))
