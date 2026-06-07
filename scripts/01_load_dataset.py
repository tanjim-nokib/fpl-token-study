# ============================================================
# 01_load_dataset.py
# Beyond Syntax — Dataset Loader & Validator
# Author: Tanjim Khan Nokib, University of Graz
# ============================================================

import pandas as pd
import json
import os

DATASET_PATH  = 'data/prompts_500.csv'
MANIFEST_PATH = 'data/experiment_manifest.json'

VALID_STRATEGIES    = ['Zero-Shot', 'Few-Shot', 'Chain-of-Thought']
VALID_LENGTHS       = ['Short', 'Medium', 'Long']
VALID_SPECIFICITY   = ['Abstract', 'Semi-Concrete', 'Concrete']
VALID_VOCAB         = ['Natural Language', 'Mixed', 'Programming-Heavy']
VALID_SYNTAX        = ['Imperative', 'Declarative', 'Question-Form']

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = [
        'prompt_id', 'task_name', 'task_difficulty',
        'strategy', 'length', 'specificity', 'vocabulary', 'syntax',
        'prompt_text'
    ]
    for col in required_cols:
        assert col in df.columns, f'Missing column: {col}'
    assert df['strategy'].isin(VALID_STRATEGIES).all(), 'Invalid strategy values'
    assert df['length'].isin(VALID_LENGTHS).all(),       'Invalid length values'
    assert df['specificity'].isin(VALID_SPECIFICITY).all(),'Invalid specificity values'
    assert df['vocabulary'].isin(VALID_VOCAB).all(),     'Invalid vocabulary values'
    assert df['syntax'].isin(VALID_SYNTAX).all(),        'Invalid syntax values'
    assert len(df) == 500, f'Expected 500 prompts, got {len(df)}'
    print(f'✓ Dataset validated: {len(df)} prompts loaded')
    return df

def build_manifest(df: pd.DataFrame) -> dict:
    manifest = {
        'total_prompts': len(df),
        'runs_per_prompt': 5,
        'total_api_calls': len(df) * 5,
        'model': 'gpt-4-turbo-2024-04-09',
        'temperature': 0.0,
        'max_tokens': 256,
        'prompts': df.to_dict(orient='records')
    }
    return manifest

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/raw', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    df = load_and_validate(DATASET_PATH)
    manifest = build_manifest(df)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'✓ Manifest saved to {MANIFEST_PATH}')
    print(f'  Breakdown by strategy:')
    print(df['strategy'].value_counts().to_string())
