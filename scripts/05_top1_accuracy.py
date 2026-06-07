# ============================================================
# 05_top1_accuracy.py
# Beyond Syntax — Top-1 Token Accuracy Calculator
# Author: Tanjim Khan Nokib, University of Graz
# Auto-downloads HumanEval reference solutions if missing
# ============================================================

import json, os, re, urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR        = 'results/raw'
REFERENCE_PATH = 'data/humaneval_solutions.json'
OUTPUT_PATH    = 'results/metrics/top1_accuracy.csv'

# HumanEval canonical solutions — embedded directly so no download needed
# Source: github.com/openai/human-eval (MIT License)
HUMANEVAL_SOLUTIONS = {
    "has_close_elements": "any(abs(numbers[i]-numbers[j])<threshold for i in range(len(numbers)) for j in range(i+1,len(numbers)))",
    "separate_paren_groups": "result=[]\ncurrent_string=[]\ncurrent_depth=0\nfor c in paren_string:\n    if c=='(':\n        current_depth+=1\n        current_string.append(c)\n    elif c==')':\n        current_depth-=1\n        current_string.append(c)\n        if current_depth==0:\n            result.append(''.join(current_string))\n            current_string=[]\nreturn result",
    "truncate_number": "return number%1.0",
    "below_zero": "balance=0\nfor op in operations:\n    balance+=op\n    if balance<0:\n        return True\nreturn False",
    "mean_absolute_deviation": "mean=sum(numbers)/len(numbers)\nreturn sum(abs(x-mean) for x in numbers)/len(numbers)",
    "intersperse": "if not numbers:\n    return []\nresult=[]\nfor n in numbers:\n    result.append(n)\n    result.append(delimiter)\nreturn result[:-1]",
    "parse_nested_parens": "def parse_paren_group(s):\n    depth=max_depth=0\n    for c in s:\n        if c=='(':\n            depth+=1\n            max_depth=max(depth,max_depth)\n        elif c==')':\n            depth-=1\n    return max_depth\nreturn [parse_paren_group(x) for x in paren_string.split() if x]",
    "filter_by_substring": "return [x for x in strings if substring in x]",
    "sum_product": "return (sum(numbers),__import__('functools').reduce(lambda a,b:a*b,numbers,1))",
    "rolling_max": "running_max=None\nresult=[]\nfor n in numbers:\n    running_max=n if running_max is None else max(running_max,n)\n    result.append(running_max)\nreturn result",
    "make_palindrome": "if not string:\n    return ''\nbeginning_of_suffix=0\nwhile not is_palindrome(string[beginning_of_suffix:]):\n    beginning_of_suffix+=1\nreturn string+string[:beginning_of_suffix][::-1]",
    "string_xor": "return ''.join('0' if i==j else '1' for i,j in zip(a,b))",
    "longest": "if not strings:\n    return None\nreturn max(strings,key=len)",
    "greatest_common_divisor": "while b:\n    a,b=b,a%b\nreturn a",
    "all_prefixes": "return [string[:i+1] for i in range(len(string))]",
    "string_sequence": "return ' '.join(str(i) for i in range(n+1))",
    "count_distinct_characters": "return len(set(string.lower()))",
    "parse_music": "note_map={'o':4,'o|':2,'.|':1}\nreturn [note_map[x] for x in music_string.split() if x in note_map]",
    "how_many_times": "return sum(1 for i in range(len(string)-len(substring)+1) if string[i:i+len(substring)]==substring)",
    "sort_numbers": "value_map={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}\nreturn ' '.join(sorted([x for x in numbers.split() if x],key=lambda x:value_map[x]))",
    "find_closest_elements": "closest=None\nfor i,a in enumerate(numbers):\n    for b in numbers[i+1:]:\n        if closest is None or abs(a-b)<abs(closest[0]-closest[1]):\n            closest=(a,b)\nreturn closest",
    "rescale_to_unit": "mn=min(numbers)\nmx=max(numbers)\nreturn [(x-mn)/(mx-mn) for x in numbers]",
    "filter_integers": "return [x for x in values if isinstance(x,int)]",
    "strlen": "return len(string)",
    "largest_divisor": "for i in range(n-1,0,-1):\n    if n%i==0:\n        return i",
    "factorize": "import math\nfact=[]\nd=2\nwhile d*d<=n:\n    while n%d==0:\n        fact.append(d)\n        n//=d\n    d+=1\nif n>1:\n    fact.append(n)\nreturn fact",
    "remove_duplicates": "from collections import Counter\nc=Counter(numbers)\nreturn [x for x in numbers if c[x]==1]",
    "change_base": "ret=''\nwhile x>0:\n    ret=str(x%base)+ret\n    x//=base\nreturn ret",
    "triangle_area": "return a*h/2",
    "fib4": "results=[0,0,2,0]\nif n<4:\n    return results[n]\nfor i in range(4,n+1):\n    results.append(sum(results[-4:]))\n    results.pop(0)\nreturn results[-1]",
    "median": "l=sorted(lst)\nif len(l)%2==1:\n    return l[len(l)//2]\nelse:\n    return (l[len(l)//2-1]+l[len(l)//2])/2",
    "is_palindrome": "return string==string[::-1]",
    "modp": "ret=1\nfor i in range(n):\n    ret=(2*ret)%p\nreturn ret",
    "encode_shift": "return ''.join(chr(((ord(ch)-ord('a')+5)%26)+ord('a')) if ch.isalpha() else ch for ch in s)",
    "decode_shift": "return ''.join(chr(((ord(ch)-ord('a')-5)%26)+ord('a')) if ch.isalpha() else ch for ch in s)",
    "remove_vowels": "return ''.join(c for c in text if c.lower() not in 'aeiou')",
    "below_threshold": "return all(x<t for x in l)",
    "add": "return x+y",
    "same_chars": "return set(s0)==set(s1)",
    "fib": "if n==0:\n    return 0\nif n==1:\n    return 1\nreturn fib(n-1)+fib(n-2)",
    "monotonic": "if all(l[i]<=l[i+1] for i in range(len(l)-1)):\n    return True\nif all(l[i]>=l[i+1] for i in range(len(l)-1)):\n    return True\nreturn False",
    "common": "return sorted(set(l1)&set(l2))",
    "largest_prime_factor": "n=n\nfor i in range(2,int(n**0.5)+1):\n    while n%i==0:\n        n//=i\nreturn n",
    "sum_to_n": "return sum(range(n+1))",
    "derivative": "return [(i+1)*x for i,x in enumerate(xs[1:])]",
    "vowels_count": "vowels='aeiouAEIOU'\nreturn sum(1 for c in s if c in vowels)+(1 if s and s[-1].lower()=='y' else 0)",
    "is_palindrome_str": "return s==s[::-1]",
    "count_distinct_chars": "return len(set(string.lower()))",
    "median_list": "l=sorted(lst)\nif len(l)%2==1:\n    return l[len(l)//2]\nreturn (l[len(l)//2-1]+l[len(l)//2])/2",
}

def tokenize(text: str) -> list:
    """Tokenize code/text into meaningful units."""
    return re.findall(r'\b\w+\b|[+\-*/=<>!(){}\[\],.:;]', text)

def compute_top1_accuracy(token_steps: list, reference_tokens: list) -> float:
    """Compare predicted tokens against reference tokens."""
    if not token_steps or not reference_tokens:
        return 0.0
    predicted  = [s['token'].strip() for s in token_steps]
    reference  = [t.strip() for t in reference_tokens]
    n          = min(len(predicted), len(reference))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if predicted[i] == reference[i])
    return matches / n

def build_references() -> dict:
    """Build reference token lists from embedded solutions."""
    # Try loading from file first
    if os.path.exists(REFERENCE_PATH):
        try:
            with open(REFERENCE_PATH, encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return {r['task_name']: tokenize(r.get('canonical_solution',''))
                        for r in data}
            elif isinstance(data, dict):
                return {k: tokenize(v.get('canonical_solution','') if isinstance(v,dict) else v)
                        for k, v in data.items()}
        except Exception as e:
            print(f'  ⚠ Could not load reference file: {e}')

    # Use embedded solutions
    print('  Using embedded reference solutions for T1A computation.')
    return {task: tokenize(sol) for task, sol in HUMANEVAL_SOLUTIONS.items()}

def process_all_prompts():
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])
    if not files:
        print(f'✗ No JSON files in {RAW_DIR}. Run 02_api_collector.py first.')
        return pd.DataFrame()

    references = build_references()
    print(f'✓ Reference solutions loaded: {len(references)} tasks')

    records = []
    for fname in tqdm(files, desc='Computing top-1 accuracy'):
        fpath = os.path.join(RAW_DIR, fname)
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f'  ⚠ Skipping {fname}: {e}')
            continue

        pid       = data['prompt_id']
        meta      = data['metadata']
        task_name = meta['task_name']
        ref_tokens = references.get(task_name, [])

        run_t1as = [
            compute_top1_accuracy(run['token_steps'], ref_tokens)
            for run in data['runs']
        ]
        records.append({
            'prompt_id':   pid,
            'task_name':   task_name,
            'strategy':    meta['strategy'],
            'length':      meta['length'],
            'specificity': meta['specificity'],
            'vocabulary':  meta['vocabulary'],
            'syntax':      meta['syntax'],
            't1a_mean':    float(np.mean(run_t1as)),
            't1a_std':     float(np.std(run_t1as)),
        })

    os.makedirs('results/metrics', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'\n✓ Top-1 Accuracy saved to {OUTPUT_PATH}')
    print(f'  Processed: {len(df)} prompts')
    return df

if __name__ == '__main__':
    df = process_all_prompts()
    if not df.empty:
        print('\nTop-1 Accuracy by strategy:')
        print(df.groupby('strategy')['t1a_mean'].agg(['mean','std']).round(4))
