# 08_merge_results.py — Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
import pandas as pd, os

def merge_all_metrics():
    metrics_dir = 'results/metrics'
    files = {
        'entropy':     'entropy.csv',
        'perplexity':  'perplexity.csv',
        'top1':        'top1_accuracy.csv',
        'consistency': 'consistency.csv',
        'ctr':         'ctr.csv',
    }
    missing = [f for f in files.values() if not os.path.exists(f'{metrics_dir}/{f}')]
    if missing:
        print(f'✗ Missing metric files: {missing}')
        print('  Run scripts 03-07 first.')
        return None

    entropy     = pd.read_csv(f'{metrics_dir}/entropy.csv')
    perplexity  = pd.read_csv(f'{metrics_dir}/perplexity.csv')
    top1        = pd.read_csv(f'{metrics_dir}/top1_accuracy.csv')
    consistency = pd.read_csv(f'{metrics_dir}/consistency.csv')
    ctr         = pd.read_csv(f'{metrics_dir}/ctr.csv')

    master = entropy[['prompt_id','task_name','strategy','length',
                      'specificity','vocabulary','syntax',
                      'entropy_mean','entropy_std']]
    master = master.merge(perplexity[['prompt_id','ppl_mean','ppl_std']], on='prompt_id')
    master = master.merge(top1[['prompt_id','t1a_mean','t1a_std']],       on='prompt_id')
    master = master.merge(consistency[['prompt_id','sc_mean']],            on='prompt_id')
    master = master.merge(ctr[['prompt_id','ctr_mean','ctr_std']],         on='prompt_id')

    # Ordinal encodings for regression
    master['strategy_code']    = master['strategy'].map({'Zero-Shot':0,'Few-Shot':1,'Chain-of-Thought':2})
    master['length_code']      = master['length'].map({'Short':0,'Medium':1,'Long':2})
    master['specificity_code'] = master['specificity'].map({'Abstract':0,'Semi-Concrete':1,'Concrete':2})
    master['vocab_code']       = master['vocabulary'].map({'Natural Language':0,'Mixed':1,'Programming-Heavy':2})
    master['syntax_code']      = master['syntax'].map({'Imperative':0,'Declarative':1,'Question-Form':2})

    master.to_csv('results/master_results.csv', index=False)
    print(f'✓ Master results saved: {len(master)} rows, {len(master.columns)} columns')
    print(master[['entropy_mean','ppl_mean','t1a_mean','sc_mean','ctr_mean']].describe().round(4))
    return master

if __name__ == '__main__':
    df = merge_all_metrics()
