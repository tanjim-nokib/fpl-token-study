# 09_statistical_analysis.py — Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
import pandas as pd, numpy as np, os
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

os.makedirs('results/stats', exist_ok=True)

DVS      = ['entropy_mean','ppl_mean','t1a_mean','sc_mean','ctr_mean']
IVS      = ['strategy_code','length_code','specificity_code','vocab_code','syntax_code']
IV_NAMES = ['strategy','length','specificity','vocabulary','syntax']

def load_data():
    path = 'results/master_results.csv'
    if not os.path.exists(path):
        print('✗ master_results.csv not found. Run 08_merge_results.py first.')
        return None
    df = pd.read_csv(path)
    print(f'✓ Loaded {len(df)} rows')
    return df

def run_manova(df):
    print('\n── MANOVA ─────────────────────────────')
    try:
        dv_str = ' + '.join(DVS)
        iv_str = ' + '.join(IVS)
        maov   = MANOVA.from_formula(f'{dv_str} ~ {iv_str}', data=df)
        result = maov.mv_test()
        with open('results/stats/manova_results.txt','w') as f:
            f.write(str(result.summary()))
        print('✓ MANOVA results saved')
    except Exception as e:
        print(f'⚠ MANOVA failed: {e}')

def run_anovas(df):
    print('\n── One-Way ANOVAs ─────────────────────')
    rows = []
    for iv, iv_name in zip(IVS, IV_NAMES):
        for dv in DVS:
            groups = [g[dv].values for _,g in df.groupby(iv)]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            grand_mean = df[dv].mean()
            ss_between = sum(len(g)*((g[dv].mean()-grand_mean)**2) for _,g in df.groupby(iv))
            ss_total   = ((df[dv]-grand_mean)**2).sum()
            eta_sq     = ss_between/ss_total if ss_total > 0 else 0
            rows.append({'IV':iv_name,'DV':dv,'F':round(F,3),'p':round(p,4),
                         'eta_sq':round(eta_sq,4),
                         'sig':'***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'})
    anova_df = pd.DataFrame(rows)
    anova_df.to_csv('results/stats/anova_results.csv', index=False)
    print(anova_df.to_string())
    print('✓ ANOVA results saved')
    return anova_df

def run_regression(df):
    print('\n── Multiple Regression ────────────────')
    rows = []
    for dv in DVS:
        formula = f'{dv} ~ {" + ".join(IVS)}'
        try:
            model = smf.ols(formula, data=df).fit()
            for param, coef, pval in zip(model.params.index, model.params, model.pvalues):
                rows.append({'DV':dv,'Predictor':param,'Beta':round(coef,4),
                             'p':round(pval,4),'R2':round(model.rsquared,4)})
            print(f'  {dv}: R²={model.rsquared:.4f}')
        except Exception as e:
            print(f'  ⚠ Regression failed for {dv}: {e}')
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv('results/stats/regression_results.csv', index=False)
    print('✓ Regression results saved')
    return reg_df

def run_correlations(df):
    print('\n── Pearson Correlations ───────────────')
    corr = df[DVS].corr(method='pearson')
    corr.to_csv('results/stats/correlation_matrix.csv')
    print(corr.round(3))
    print('✓ Correlation matrix saved')

def run_tukey(df):
    print('\n── Tukey HSD Post-Hoc ─────────────────')
    rows = []
    for iv_name in IV_NAMES:
        for dv in DVS:
            try:
                result = pairwise_tukeyhsd(endog=df[dv], groups=df[iv_name], alpha=0.05)
                for row in result.summary().data[1:]:
                    rows.append({'IV':iv_name,'DV':dv,'group1':row[0],'group2':row[1],
                                 'meandiff':row[2],'p_adj':row[3],'reject':row[6]})
            except Exception as e:
                print(f'  ⚠ Tukey failed for {iv_name}/{dv}: {e}')
    tukey_df = pd.DataFrame(rows)
    tukey_df.to_csv('results/stats/tukey_results.csv', index=False)
    print('✓ Tukey HSD results saved')

if __name__ == '__main__':
    df = load_data()
    if df is not None:
        run_manova(df)
        run_anovas(df)
        run_regression(df)
        run_correlations(df)
        run_tukey(df)
        print('\n✓ All statistical analyses complete.')
        print('  Next step: python 10_visualizations.py')
