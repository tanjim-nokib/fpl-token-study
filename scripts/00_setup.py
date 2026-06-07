# ============================================================
# 00_setup.py
# Beyond Syntax — Environment Setup & API Verification
# Author: Tanjim Khan Nokib, University of Graz
# ============================================================

import subprocess, sys, os

REQUIRED_PACKAGES = [
    'groq', 'numpy', 'scipy', 'statsmodels', 'scikit-learn',
    'pandas', 'matplotlib', 'seaborn', 'sentence-transformers',
    'tiktoken', 'pingouin', 'tqdm', 'python-dotenv',
]

def install_packages():
    print('Installing required packages...')
    for pkg in REQUIRED_PACKAGES:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
        print(f'  ✓ {pkg}')

def verify_api():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print('\n✗ GROQ_API_KEY not found in .env file.')
        print('  Add this line to your .env file:')
        print('  GROQ_API_KEY=gsk_your-key-here')
        sys.exit(1)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',   # current active model
            messages=[{'role': 'user', 'content': 'Say OK in one word.'}],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        print('\n✓ Groq API verified successfully!')
        print(f'  Model: llama-3.3-70b-versatile (free, 14,400 req/day)')
        print(f'  Test response: {answer}')
    except Exception as e:
        print(f'\n✗ Groq API verification failed: {e}')
        sys.exit(1)

def check_folders():
    for folder in ['data','results/raw','results/metrics','results/stats','figures','logs']:
        os.makedirs(folder, exist_ok=True)
    print('✓ All project folders ready.')

def check_dataset():
    path = 'data/prompts_500.csv'
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        print(f'✓ Dataset found: {len(df)} prompts loaded.')
        missing = df[df['prompt_text'].str.contains(r'\[INSERT', na=False)]
        if len(missing) > 0:
            print(f'  ⚠ {len(missing)} rows still have placeholder text.')
        else:
            print(f'  ✓ All 500 prompt texts filled in correctly.')
    else:
        print('⚠ data/prompts_500.csv not found.')
        print('  Download prompts_500.csv from Claude and place it in data/ folder.')

if __name__ == '__main__':
    print('=' * 55)
    print('  Beyond Syntax — Setup Verification')
    print('  Tanjim Khan Nokib | University of Graz')
    print('=' * 55)
    install_packages()
    verify_api()
    check_folders()
    check_dataset()
    print('\n' + '=' * 55)
    print('✓ Setup complete! Next: python 01_load_dataset.py')
    print('=' * 55)
