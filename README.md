Kaggle Recommender-Systems Competition

**Final score:** Recall\@10 = 0.0448

A tier-specific hybrid recommender that topped a sparse-data challenge with
330 K users, 65 K items and just 350 K interactions (94 % cold-start).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Innovations](#key-innovations)
4. [Performance Results](#performance-results)
5. [Technical Deep Dive](#technical-deep-dive)
6. [Implementation Details](#implementation-details)
7. [Repository Structure](#repository-structure)
8. [Key Learnings](#key-learnings)
9. [Future Directions](#future-directions)
10. [Business Impact](#business-impact)

---

## Project Overview <a name="project-overview"></a>

The goal: generate accurate top-10 personalised recommendations for users with *extremely* limited history.

* **94.6 %** of users have only 1 interaction
* **Extreme sparsity:** 350 K interactions across 330 K users / 65 K items
* **Real-world relevance:** typical e-commerce cold-start scenario

Our solution fuses several paradigms in an adaptive, tier-specific pipeline.

---

## System Architecture <a name="system-architecture"></a>

### High‑level view

![System architecture](https://github.com/user-attachments/assets/3f3d2cc5-8907-4b98-80f9-5dc33bedc4c1)

### Modular hybrid stack

| Module                     | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| Sequential pattern mining  | 1‑hop / 2‑hop transition matrices (Laplace α = 0.1) |
| Content‑based retrieval    | Sentence‑BERT on enriched metadata                  |
| Collaborative filtering    | Item‑item Jaccard for popular items                 |
| Bought‑together extraction | Three‑layer co‑purchase mining                      |
| Matrix factorisation       | Truncated SVD (50 latent factors)                   |
| Smart popularity           | Time‑decayed trending with burst detection          |

### Tier‑specific strategy

| Tier                 | Users  | Strategy focus       | Key modules                      |
| -------------------- | ------ | -------------------- | -------------------------------- |
| Cold (1 interaction) | 94.6 % | Content + Popularity | Enriched embeddings, metadata    |
| Warm (2‑4)           | 5.3 %  | Sequential + Content | Pattern mining, balanced weights |
| Hot (5+)             | 0.1 %  | Collaborative + Seq. | Full feature set, regularisation |

---

## Key Innovations <a name="key-innovations"></a>

1. **Enhanced content similarity** – concatenate *title + details + category*; Sentence‑BERT embeddings, 30 000 TF‑IDF bi‑gram fallback.
2. **Bought‑together mining** – session, user and sequential co‑occurrence; weighted fusion.
3. **Combo bonus** – +25 % score for candidates surfacing in ≥ 2 modules.
4. **Adaptive weighting** – module weights learned per tier via Bayesian optimisation.

---

## Performance Results <a name="performance-results"></a>

* **Final Recall\@10:** 0.0426
* **Improvement over popularity‑only:** +119 % (baseline = 0.0181)

| Tier | Recall\@10 | Std Dev | Population |
| ---- | ---------- | ------- | ---------- |
| Cold | 0.0422     | 0.0067  | 94.6 %     |
| Warm | 0.0323     | 0.0084  | 5.3 %      |
| Hot  | 0.0294     | 0.0239  | 0.1 %      |

### Visual summary

![Tier-Specific Recall](https://github.com/user-attachments/assets/1a1efbe3-ee0d-4a59-be5a-b79217d2b123)


---

## Technical Deep Dive <a name="technical-deep-dive"></a>

* **Data preprocessing:** deduplication, temporal sorting, metadata enrichment, Wilson quality scores.
* **Hyperparameter search:** Optuna (40 trials, 24 params), multi‑objective weighting by user tier.
* **Regularisation:** Laplace smoothing, diversity penalties, tier‑aware candidate caps.

```python
wilson = (
    p_hat + z**2 / (2 * n)
    - z * sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
) / (1 + z**2 / n)

quality = wilson + log1p(count) / 10
```

---

## Implementation Details <a name="implementation-details"></a>

| Aspect               | Value                                                                         |
| -------------------- | ----------------------------------------------------------------------------- |
| Memory footprint     | \~2.3 GB (pre‑computed structures)                                            |
| Inference throughput | \~100 recs/s on standard CPU                                                  |
| Tech stack           | Python 3.8, pandas, NumPy, scikit‑learn, sentence‑transformers, SciPy, Optuna |

---

## Repository Structure <a name="repository-structure"></a>

```text
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── item_meta.csv
│   └── item_meta_enhanced.csv
├── src/
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── pattern_mining.py
│   └── optimization.py
├── models/
│   └── trained_models/
├── results/
│   ├── submission.csv
│   └── analysis/
└── notebooks/
    ├── EDA.ipynb
    └── evaluation.ipynb
```

---

## Key Learnings <a name="key-learnings"></a>

* Tier‑specific adaptation yields large gains across segments.
* Multi‑signal fusion adds robustness to sparse signals.
* Metadata enrichment is vital for cold‑start performance.
* Bayesian optimisation surfaces non‑obvious but impactful configs.

---

## Future Directions <a name="future-directions"></a>

* Meta‑learning for few‑shot recommendation
* Graph neural networks for dynamic relationships
* Contextual bandits for exploration–exploitation
* Continuous tier modelling with smooth weighting

---

## Business Impact <a name="business-impact"></a>

* Higher engagement from relevant recommendations
* Architecture scales to real‑time production
* Cold‑start onboarding for new users/items
* Adaptable framework for diverse behaviour patterns

---

*Project completed for the Recommender Systems course, Leiden University (Spring 2025).*
