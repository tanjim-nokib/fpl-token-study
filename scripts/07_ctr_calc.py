# 07_ctr_calc.py — Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
import json, os, ast, re, keyword
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR     = 'results/raw'
OUTPUT_PATH = 'results/metrics/ctr.csv'

PYTHON_KEYWORDS = set(keyword.kwlist)
CODE_OPERATORS  = set(['+','-','*','/','//','%','**','=','==','!=','<','>','<=','>='])
CODE_DELIMITERS = set(['(',')','{','}','[',']',':',',','.','@','#'])
CODE_PATTERN    = re.compile(r'^(def |class |import |from |return|if |else|elif |for |while |try|except|with |lambda|yield|pass|break|continue|None|True|False|\d+\.?\d*|#.*$)')

def classify_token(token: str) -> str:
    t = token.strip()
    if not t: return 'natural_language'
    if t in PYTHON_KEYWORDS: return 'code'
    if t in CODE_OPERATORS or t in CODE_DELIMITERS: return 'code'
    if CODE_PATTERN.match(t): return 'code'
    try:
        ast.parse(t)
        return 'code'
    except SyntaxError:
        pass
    return 'natural_language'

def compute_ctr(token_steps: list) -> float:
    if not token_steps: return 0.0
    labels = [classify_token(s['token']) for s in token_steps]
    return labels.count('code') / len(labels)

def process_all_prompts():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])
    if not files:
        print(f'✗ No JSON files in {RAW_DIR}. Run 02_api_collector.py first.')
        return pd.DataFrame()
    records = []
    for fname in tqdm(files, desc='Computing CTR'):
        with open(os.path.join(RAW_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        pid  = data['prompt_id']
        meta = data['metadata']
        run_ctrs = [compute_ctr(run['token_steps']) for run in data['runs']]
        records.append({
            'prompt_id':   pid,
            'task_name':   meta['task_name'],
            'strategy':    meta['strategy'],
            'length':      meta['length'],
            'specificity': meta['specificity'],
            'vocabulary':  meta['vocabulary'],
            'syntax':      meta['syntax'],
            'ctr_mean':    float(np.mean(run_ctrs)),
            'ctr_std':     float(np.std(run_ctrs)),
        })
    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ CTR saved to {OUTPUT_PATH}')
    print(f'  Mean CTR: {df["ctr_mean"].mean():.4f}')
    return df

if __name__ == '__main__':
    df = process_all_prompts()
    if not df.empty:
        print(df.groupby('vocabulary')['ctr_mean'].agg(['mean','std']).round(4))
