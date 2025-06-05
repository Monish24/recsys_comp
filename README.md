
### Modular hybrid stack

| Module                              | Purpose                                             |
| ----------------------------------- | --------------------------------------------------- |
| Sequential pattern mining           | 1-hop / 2-hop transition matrices (Laplace α = 0.1) |
| Content-based retrieval             | Sentence-BERT on enriched metadata                 |
| Collaborative filtering             | Item-item Jaccard for popular items                |
| Bought-together extraction          | Three-layer co-purchase mining                     |
| Matrix factorisation                | Truncated SVD (50 latent factors)                  |
| Smart popularity                    | Time-decayed trending with burst detection         |

### Tier-specific strategy

| Tier | Users | Strategy focus           | Key modules                            |
| ---- | ----- | ------------------------ | -------------------------------------- |
| Cold (1 interaction) | 94.6 % | Content + Popularity | Enriched embeddings, metadata |
| Warm (2-4)           | 5.3 %  | Sequential + Content | Pattern mining, balanced weights |
| Hot (5+)             | 0.1 %  | Collaborative + Seq. | Full feature set, regularisation |

---

## Key Innovations <a name="key-innovations"></a>

1. **Enhanced content similarity** – concatenate *title + details + category*; Sentence-BERT embeddings, 30 000 TF-IDF bi-gram fallback.  
2. **Bought-together mining** – session, user and sequential co-occurrence; weighted fusion.  
3. **Combo bonus** – +25 % score for candidates surfacing in ≥ 2 modules.  
4. **Adaptive weighting** – module weights learned per tier via Bayesian optimisation.

---

## Performance Results <a name="performance-results"></a>

* **Final Recall@10:** 0.0426  
* **Improvement over popularity-only:** +119 % (baseline = 0.0181)  

| Tier | Recall@10 | Std Dev | Population |
| ---- | --------- | ------ | ---------- |
| Cold | 0.0422    | 0.0067 | 94.6 % |
| Warm | 0.0323    | 0.0084 | 5.3 % |
| Hot  | 0.0294    | 0.0239 | 0.1 % |

### Visual summary


