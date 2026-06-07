# 10_visualizations.py — Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('figures', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = ['#1F3864','#2E5496','#5B9BD5','#9DC3E6','#DEEBF7']

def load():
    path = 'results/master_results.csv'
    if not os.path.exists(path):
        print('✗ master_results.csv not found. Run 08_merge_results.py first.')
        return None
    return pd.read_csv(path)

def fig2_entropy_by_strategy(df):
    fig, ax = plt.subplots(figsize=(8,5))
    order = ['Zero-Shot','Few-Shot','Chain-of-Thought']
    means = df.groupby('strategy')['entropy_mean'].mean().reindex(order)
    sems  = df.groupby('strategy')['entropy_mean'].sem().reindex(order)
    bars  = ax.bar(order, means, yerr=sems, capsize=5,
                   color=PALETTE[:3], edgecolor='white', linewidth=0.8)
    ax.set_xlabel('Prompting Strategy', fontsize=12)
    ax.set_ylabel('Mean Token Entropy (bits)', fontsize=12)
    ax.set_title('Figure 2: Token Entropy by Prompting Strategy (±SE)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig2_entropy_by_strategy.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2_entropy_by_strategy.pdf', bbox_inches='tight')
    plt.close()
    print('✓ Figure 2 saved')

def fig3_correlation_heatmap(df):
    dvs    = ['entropy_mean','ppl_mean','t1a_mean','sc_mean','ctr_mean']
    labels = ['Entropy','Perplexity','Top-1 Acc','Consistency','Code Ratio']
    corr   = df[dvs].corr()
    corr.index = corr.columns = labels
    fig, ax = plt.subplots(figsize=(7,6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
                cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Figure 3: Pearson Correlation Matrix — Dependent Variables',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Figure 3 saved')

def fig4_regression_coefs(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    iv_codes  = ['strategy_code','length_code','specificity_code','vocab_code','syntax_code']
    iv_labels = ['Strategy','Length','Specificity','Vocabulary','Syntax']
    X  = df[iv_codes].values
    Xs = StandardScaler().fit_transform(X)
    fig, axes = plt.subplots(1,5,figsize=(15,4),sharey=True)
    dv_labels = ['Entropy','Perplexity','T1 Accuracy','Consistency','Code Ratio']
    for ax, dv, label in zip(axes,
        ['entropy_mean','ppl_mean','t1a_mean','sc_mean','ctr_mean'], dv_labels):
        y     = df[dv].values
        model = LinearRegression().fit(Xs, y)
        coefs = model.coef_
        colors= ['#D85A30' if c>0 else '#1F3864' for c in coefs]
        ax.barh(iv_labels, coefs, color=colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Std. β', fontsize=9)
    fig.suptitle('Figure 4: Standardized Regression Coefficients per Dependent Variable',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig4_regression_coefs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Figure 4 saved')

def fig5_interaction(df):
    fig, ax = plt.subplots(figsize=(8,5))
    strategies = ['Zero-Shot','Few-Shot','Chain-of-Thought']
    vocabs     = ['Natural Language','Mixed','Programming-Heavy']
    for strat, color in zip(strategies, PALETTE):
        means = [df[(df['strategy']==strat)&(df['vocabulary']==v)]['entropy_mean'].mean()
                 for v in vocabs]
        ax.plot(vocabs, means, marker='o', label=strat, color=color, linewidth=2)
    ax.set_xlabel('Domain Vocabulary', fontsize=12)
    ax.set_ylabel('Mean Token Entropy (bits)', fontsize=12)
    ax.set_title('Figure 5: Interaction — Prompting Strategy × Domain Vocabulary on Entropy',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig('figures/fig5_interaction_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Figure 5 saved')

if __name__ == '__main__':
    df = load()
    if df is not None:
        fig2_entropy_by_strategy(df)
        fig3_correlation_heatmap(df)
        fig4_regression_coefs(df)
        fig5_interaction(df)
        print('\n✓ All figures saved in figures/ folder')
        print('  Next step: python 11_fpl_axiom_mapper.py')
