# FPL Token Study

**Beyond Syntax: Token-Level Evidence for a New Generation of Human–LLM Programming Languages**

**Author:** Tanjim Khan Nokib
**Institution:** Karl-Franzens-Universität Graz, Computational Social Systems, Austria
**Submitted to:** ACM Transactions on Programming Languages and Systems (TOPLAS)
**arXiv:** [To be added]

---

## Overview

This repository contains all code, data, and results for the paper.
We conducted a controlled quantitative experiment varying five prompt
design variables across 500 HumanEval prompts submitted to LLaMA 3.3 70B
via the Groq API (temperature = 0.0, 2,500 total API calls), measuring
five token-level output metrics.

**Key finding:** Few-Shot prompts with programming-heavy vocabulary
produced the lowest output entropy (M = 4.326 bits), translating into
five design axioms for a Future Programmer Language (FPL).

---

## Repository Structure

```
fpl-token-study/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/                        ← Full 12-script pipeline
│   ├── 00_setup.py                 ← Install packages, verify API
│   ├── 01_load_dataset.py          ← Load and validate 500 prompts
│   ├── 02_api_collector.py         ← Groq API data collection (2,500 calls)
│   ├── 03_entropy_calc.py          ← Shannon entropy computation
│   ├── 04_perplexity_calc.py       ← Proxy perplexity computation
│   ├── 05_top1_accuracy.py         ← Top-1 token accuracy
│   ├── 06_consistency_calc.py      ← Semantic consistency (TF-IDF)
│   ├── 07_ctr_calc.py              ← Code token ratio (AST parsing)
│   ├── 08_merge_results.py         ← Merge all metrics into master CSV
│   ├── 09_statistical_analysis.py  ← MANOVA, ANOVA, regression, Tukey HSD
│   ├── 10_visualizations.py        ← Generate Figures 2–5
│   └── 11_fpl_axiom_mapper.py      ← Map results to FPL design axioms
├── data/
│   └── prompts_500.csv             ← 500-prompt benchmark dataset
├── results/
│   ├── master_results.csv          ← All 500 prompts × 5 metrics
│   ├── metrics/                    ← Individual metric CSV files
│   │   ├── entropy.csv
│   │   ├── perplexity.csv
│   │   ├── top1_accuracy.csv
│   │   ├── consistency.csv
│   │   └── ctr.csv
│   └── stats/                      ← Statistical output tables
│       ├── anova_results.csv
│       ├── regression_results.csv
│       ├── correlation_matrix.csv
│       └── tukey_results.csv
└── figures/
    ├── fig1_conceptual_framework.png  ← Conceptual framework (Figure 1)
    ├── fig2_entropy_by_strategy.png   ← Entropy by strategy (Figure 2)
    ├── fig3_correlation_heatmap.png   ← Correlation heatmap (Figure 3)
    ├── fig4_regression_coefs.png      ← Regression coefficients (Figure 4)
    └── fig5_interaction_plot.png      ← Interaction plot (Figure 5)
```

---

## Quick Start

### Step 1 — Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) → Sign up free →
Create API Key

### Step 2 — Set up environment
```bash
git clone https://github.com/tanjimkhannokib/fpl-token-study.git
cd fpl-token-study
pip install -r requirements.txt
echo "GROQ_API_KEY=your-key-here" > .env
```

### Step 3 — Run the pipeline
```bash
python scripts/00_setup.py
python scripts/01_load_dataset.py
python scripts/02_api_collector.py      # ~2 hours, fully resumable
python scripts/03_entropy_calc.py
python scripts/04_perplexity_calc.py
python scripts/05_top1_accuracy.py
python scripts/06_consistency_calc.py
python scripts/07_ctr_calc.py
python scripts/08_merge_results.py
python scripts/09_statistical_analysis.py
python scripts/10_visualizations.py
python scripts/11_fpl_axiom_mapper.py
```

---

## Dataset

**`data/prompts_500.csv`** — 500 prompts derived from the
[HumanEval benchmark](https://github.com/openai/human-eval)
(Chen et al., 2021), systematically varying five independent variables:

| Variable | Levels |
|---|---|
| Prompt Length | Short / Medium / Long |
| Prompt Specificity | Abstract / Semi-Concrete / Concrete |
| Prompting Strategy | Zero-Shot / Few-Shot / Chain-of-Thought |
| Domain Vocabulary | Natural Language / Mixed / Programming-Heavy |
| Syntactic Structure | Imperative / Declarative |

Full raw API outputs (500 JSON files, ~50 MB) are available on Zenodo:
**DOI:** [To be registered upon paper acceptance]

---

## Key Results

| Metric | Mean | SD | Min | Max |
|---|---|---|---|---|
| Output Entropy H (bits) | 4.369 | 0.135 | 3.822 | 4.729 |
| Proxy Perplexity (PPL) | 20.752 | 1.935 | 14.144 | 26.527 |
| Semantic Consistency (SC) | 0.969 | 0.055 | 0.664 | 1.000 |
| Code Token Ratio (CTR) | 0.906 | 0.040 | 0.700 | 0.962 |

**Strongest ANOVA effects on entropy:**
- Prompt Length: F = 11.747, p < .001, η² = .045
- Prompt Specificity: F = 10.707, p < .001, η² = .041
- Domain Vocabulary: F = 10.707, p < .001, η² = .041
- Prompting Strategy: F = 8.660, p = .0002, η² = .034

---

## FPL Design Axioms

| Axiom | Description |
|---|---|
| A1 — Example Binding | Native EXAMPLE keyword for worked demonstrations |
| A2 — Domain Namespace | DOMAIN declarations for vocabulary namespaces |
| A3 — Mandatory Specification | Typed GIVEN/RETURN declarations required |
| A4 — Sequential Control Flow | STEPS blocks as core control flow primitive |
| A5 — Imperative Default | Imperative paradigm as default mode |

---

## Model and Reproducibility

- **Model:** LLaMA 3.3 70B (`llama-3.3-70b-versatile`) via Groq API
- **Temperature:** 0.0 (fully deterministic)
- **Max tokens:** 256
- **Runs per prompt:** 5
- **Total API calls:** 2,500
- **Cost:** Free (Groq free tier)

---

## Citation

```bibtex
@article{nokib2026beyond,
  author    = {Nokib, Tanjim Khan},
  title     = {Beyond Syntax: Token-Level Evidence for a New Generation
               of Human--{LLM} Programming Languages},
  journal   = {ACM Transactions on Programming Languages and Systems},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Code: MIT License
Data: Creative Commons Attribution 4.0 (CC BY 4.0)

---

## Contact

Tanjim Khan Nokib
tanjim.nokib@edu.uni-graz.at
Karl-Franzens-Universität Graz, Austria
