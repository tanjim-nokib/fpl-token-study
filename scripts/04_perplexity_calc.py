# ============================================================
# 04_perplexity_calc.py
# Beyond Syntax — Perplexity Calculator
# Author: Tanjim Khan Nokib, University of Graz
# Computes proxy perplexity from output text compression ratio
# (Groq does not support logprobs)
# ============================================================

import json, math, os, zlib
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR     = 'results/raw'
OUTPUT_PATH = 'results/metrics/perplexity.csv'

def compute_perplexity_from_text(text: str) -> float:
    """
    Proxy perplexity using unigram character model.
    PPL = exp(H) where H is character-level entropy.
    Lower PPL = more predictable, structured output.
    Valid proxy for code generation quality assessment.
    """
    if not text or len(text) < 2:
        return 100.0
    counts = Counter(text)
    total  = len(text)
    entropy = -sum((c/total)*math.log2(c/total) for c in counts.values())
    # Convert from bits to nats, then exponentiate
    return float(2 ** entropy)

def compute_compression_ratio(text: str) -> float:
    """
    Compression ratio as additional predictability metric.
    Lower ratio = more repetitive/structured = more code-like.
    """
    if not text:
        return 1.0
    raw  = len(text.encode('utf-8'))
    comp = len(zlib.compress(text.encode('utf-8'), level=9))
    return comp / raw if raw > 0 else 1.0

def process_all_prompts():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])
    if not files:
        print(f'✗ No JSON files in {RAW_DIR}. Run 02_api_collector.py first.')
        return pd.DataFrame()

    records = []
    for fname in tqdm(files, desc='Computing perplexity'):
        with open(os.path.join(RAW_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)

        pid  = data['prompt_id']
        meta = data['metadata']

        run_ppls   = []
        run_comps  = []

        for run in data['runs']:
            text = run['generated_text']
            run_ppls.append(compute_perplexity_from_text(text))
            run_comps.append(compute_compression_ratio(text))

        records.append({
            'prompt_id':    pid,
            'task_name':    meta['task_name'],
            'strategy':     meta['strategy'],
            'length':       meta['length'],
            'specificity':  meta['specificity'],
            'vocabulary':   meta['vocabulary'],
            'syntax':       meta['syntax'],
            'ppl_mean':     float(np.mean(run_ppls)),
            'ppl_std':      float(np.std(run_ppls)),
            'comp_ratio_mean': float(np.mean(run_comps)),
        })

    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Perplexity saved to {OUTPUT_PATH}')
    print(f'  Processed: {len(df)} prompts')
    print(f'  Mean perplexity: {df["ppl_mean"].mean():.4f}')
    return df

if __name__ == '__main__':
    df = process_all_prompts()
    if not df.empty:
        print('\nPerplexity by strategy:')
        print(df.groupby('strategy')['ppl_mean'].agg(['mean','std']).round(4))
