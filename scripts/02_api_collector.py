# ============================================================
# 02_api_collector.py
# Beyond Syntax — Groq API Data Collection Pipeline
# Author: Tanjim Khan Nokib, University of Graz
# Model: llama-3.3-70b-versatile (Groq free tier)
# Adjusted delay to stay within free tier rate limits
# ============================================================

import json, os, time, logging, math
from collections import Counter
from dotenv import load_dotenv
from tqdm import tqdm
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

MODEL           = 'llama-3.3-70b-versatile'
TEMPERATURE     = 0.0
MAX_TOKENS      = 256
RUNS_PER_PROMPT = 5
# Groq free tier: ~30 req/min, ~6000 tokens/min
# 256 tokens × 5 runs = 1280 tokens per prompt
# Safe delay: 15 seconds per prompt = 4 prompts/min = ~1280 tokens/min
DELAY_BETWEEN   = 15.0

os.makedirs('logs', exist_ok=True)
os.makedirs('results/raw', exist_ok=True)

logging.basicConfig(
    filename='logs/api_collector.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def compute_char_entropy(text: str) -> float:
    if not text: return 0.0
    counts = Counter(text)
    total  = len(text)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def compute_word_entropy(text: str) -> float:
    words = text.split()
    if not words: return 0.0
    counts = Counter(words)
    total  = len(words)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def build_token_steps(text: str) -> list:
    import re
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return [{'token': t, 'logprob': -1.0, 'top_logprobs': {t: -1.0}} for t in tokens]

def call_groq(prompt_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {'role': 'system', 'content': 'You are a Python programming assistant. Write complete, correct Python functions only.'},
            {'role': 'user',   'content': prompt_text}
        ]
    )
    text = response.choices[0].message.content or ''
    return {
        'generated_text': text,
        'token_steps':    build_token_steps(text),
        'char_entropy':   compute_char_entropy(text),
        'word_entropy':   compute_word_entropy(text),
        'model':          MODEL,
        'usage': {
            'prompt_tokens':     response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }
    }

def call_with_retry(prompt_text: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            return call_groq(prompt_text)
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate' in err.lower():
                # Exponential backoff starting at 60s
                wait = 60 * (attempt + 1)
                logging.warning(f'Rate limit. Waiting {wait}s...')
                print(f'\n  ⚠ Rate limit — waiting {wait}s before retry {attempt+1}/{max_retries}...')
                time.sleep(wait)
            else:
                logging.error(f'API error: {e}')
                raise
    raise RuntimeError('Max retries exceeded.')

def collect_all(manifest_path: str = 'data/experiment_manifest.json'):
    with open(manifest_path) as f:
        manifest = json.load(f)

    prompts = manifest['prompts']
    total   = len(prompts)

    # Count already done
    already_done = sum(
        1 for p in prompts
        if os.path.exists(f'results/raw/prompt_{p["prompt_id"]:04d}.json')
    )
    remaining = total - already_done

    print(f'\nData collection status:')
    print(f'  Total prompts:     {total}')
    print(f'  Already collected: {already_done}')
    print(f'  Remaining:         {remaining}')
    print(f'  Model: {MODEL}')
    print(f'  Delay: {DELAY_BETWEEN}s between prompts (safe for free tier)')
    print(f'  Estimated time for remaining: ~{round(remaining * DELAY_BETWEEN * RUNS_PER_PROMPT / 3600, 1)} hours')
    print(f'  Resume: already-done prompts are skipped automatically.\n')

    done    = 0
    skipped = 0

    for prompt in tqdm(prompts, desc='Collecting', unit='prompt'):
        pid      = prompt['prompt_id']
        out_path = f'results/raw/prompt_{pid:04d}.json'

        if os.path.exists(out_path):
            skipped += 1
            continue

        runs = []
        success = True
        try:
            for run_idx in range(RUNS_PER_PROMPT):
                result = call_with_retry(prompt['prompt_text'])
                result['run_index'] = run_idx
                runs.append(result)
                if run_idx < RUNS_PER_PROMPT - 1:
                    time.sleep(DELAY_BETWEEN)

            # Delay between prompts
            time.sleep(DELAY_BETWEEN)

            output = {'prompt_id': pid, 'metadata': prompt, 'runs': runs}
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            done += 1
            logging.info(f'Prompt {pid} collected.')

        except Exception as e:
            logging.error(f'Prompt {pid} failed: {e}')
            print(f'\n  ✗ Prompt {pid} failed: {e}')
            print(f'  Skipping and continuing...')
            time.sleep(30)
            continue

    print(f'\n{"="*55}')
    print(f'✓ Data collection session complete!')
    print(f'  Collected this session: {done}')
    print(f'  Skipped (already done): {skipped}')
    total_done = done + skipped
    print(f'  Total collected so far: {total_done} / {total}')
    if total_done < total:
        print(f'\n  ⚠ {total - total_done} prompts remaining.')
        print(f'  Run "python 02_api_collector.py" again to continue.')
    else:
        print(f'\n  ✓ All {total} prompts collected!')
        print(f'  Next step: python 03_entropy_calc.py')
    print(f'{"="*55}')

if __name__ == '__main__':
    collect_all()