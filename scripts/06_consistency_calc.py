# 06_consistency_calc.py - Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
# Uses TF-IDF instead of sentence-transformers (no PyTorch needed)

import json, os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

RAW_DIR     = 'results/raw'
OUTPUT_PATH = 'results/metrics/consistency.csv'

def compute_consistency(texts):
    if len(texts) < 2:
        return 1.0
    try:
        vec    = TfidfVectorizer(min_df=1, stop_words=None)
        matrix = vec.fit_transform(texts).toarray()
        sims   = [
            float(cosine_similarity(matrix[i].reshape(1,-1), matrix[j].reshape(1,-1))[0][0])
            for i, j in combinations(range(len(matrix)), 2)
        ]
        return float(np.mean(sims))
    except Exception:
        return 0.5

def process_all_prompts():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])
    if not files:
        print('No JSON files found. Run 02_api_collector.py first.')
        return pd.DataFrame()

    records = []
    for fname in tqdm(files, desc='Computing consistency'):
        fpath = os.path.join(RAW_DIR, fname)
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        meta  = data['metadata']
        texts = [r['generated_text'] for r in data['runs'] if r.get('generated_text','').strip()]
        sc    = compute_consistency(texts)
        records.append({
            'prompt_id':   data['prompt_id'],
            'task_name':   meta['task_name'],
            'strategy':    meta['strategy'],
            'length':      meta['length'],
            'specificity': meta['specificity'],
            'vocabulary':  meta['vocabulary'],
            'syntax':      meta['syntax'],
            'sc_mean':     sc,
        })

    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Consistency saved to {OUTPUT_PATH}')
    print(f'  Processed: {len(df)} prompts')
    print(f'  Mean SC: {df["sc_mean"].mean():.4f}')
    return df

if __name__ == '__main__':
    df = process_all_prompts()
    if not df.empty:
        print('\nConsistency by strategy:')
        print(df.groupby('strategy')['sc_mean'].agg(['mean','std']).round(4))
