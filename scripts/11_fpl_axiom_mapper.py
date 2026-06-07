# 11_fpl_axiom_mapper.py — Beyond Syntax
# Author: Tanjim Khan Nokib, University of Graz
import pandas as pd, os

AXIOM_MAP = {
    'strategy':    {'finding':'Chain-of-thought prompts produce significantly lower token entropy','axiom':'A1: FPL must support sequential step-by-step instruction structures as a core control flow primitive','pl_analogue':'Structured control flow (Dijkstra, 1968)'},
    'specificity': {'finding':'Concrete prompts with worked examples improve Top-1 accuracy significantly','axiom':'A2: FPL must include native example-binding syntax for task definitions','pl_analogue':'Function signatures with typed default arguments'},
    'vocabulary':  {'finding':'Programming-heavy vocabulary reduces entropy and raises code token ratio','axiom':'A3: FPL must include a standardized namespaced domain vocabulary layer','pl_analogue':'Import statements and type namespaces'},
    'length':      {'finding':'Medium-length prompts optimize the entropy-accuracy trade-off','axiom':'A4: FPL must enforce strong specification with mandatory context fields','pl_analogue':'Strong static typing with mandatory annotations'},
    'syntax':      {'finding':'Imperative syntactic structure maximizes semantic consistency across runs','axiom':'A5: FPL default paradigm must be imperative; declarative is an optional extension','pl_analogue':'Procedural programming as default paradigm'},
}

def generate_axiom_report():
    print('\n' + '='*65)
    print('  BEYOND SYNTAX — FPL DESIGN AXIOM REPORT')
    print('  Tanjim Khan Nokib | University of Graz | 2026')
    print('='*65)

    anova_path = 'results/stats/anova_results.csv'
    anova_df   = pd.read_csv(anova_path) if os.path.exists(anova_path) else None

    rows = []
    for iv_name, axiom_data in AXIOM_MAP.items():
        effect_str = 'See ANOVA results'
        if anova_df is not None:
            iv_results = anova_df[anova_df['IV']==iv_name].sort_values('eta_sq', ascending=False)
            if len(iv_results) > 0:
                top = iv_results.iloc[0]
                effect_str = f'η²={top["eta_sq"]:.4f} on {top["DV"]} (F={top["F"]:.2f}, {top["sig"]})'
        print(f'\n  IV: {iv_name.upper()}')
        print(f'  Empirical Finding: {axiom_data["finding"]}')
        print(f'  Effect Size:       {effect_str}')
        print(f'  FPL Axiom:         {axiom_data["axiom"]}')
        print(f'  PL Analogue:       {axiom_data["pl_analogue"]}')
        rows.append({'IV':iv_name,'Empirical Finding':axiom_data['finding'],
                     'Effect Size':effect_str,'FPL Axiom':axiom_data['axiom'],
                     'Traditional PL Analogue':axiom_data['pl_analogue']})

    print('\n' + '='*65)
    axiom_df = pd.DataFrame(rows)
    axiom_df.to_csv('results/fpl_axiom_table.csv', index=False)
    print('\n✓ FPL Axiom Table saved to results/fpl_axiom_table.csv')
    print('  Use this table as Table 6 in Section 6.2 of your paper.')
    print('\n✓ PIPELINE COMPLETE! All results ready for paper insertion.')
    return axiom_df

if __name__ == '__main__':
    generate_axiom_report()
