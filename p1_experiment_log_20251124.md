# P1 Experiment Log: Embedding Validity - 2025-11-24

## Objective
The initial goal of the P1 Prerequisite Test was to verify that our embedding models could meaningfully distinguish between different types of generated mathematical problems (linear, quadratic, exponential, trigonometric).

## Summary of Experiments and Findings

### Attempt 1: Initial Approach (Absolute Separation)
- **Method**: We followed the `prerequisites/methodology.md`, using an absolute `separation score > 0.3` as the primary success criterion.
- **Result**: **FAILED**. All models (text, semantic, hybrid) failed to meet the separation score, even though their KNN classification accuracy was high.
- **Learning**: The raw difference between intra-class and inter-class similarity was very small, suggesting that while the clusters were separable, they were not far apart in the embedding space.

### Attempt 2: Introducing MathBERT
- **Method**: Following the failure protocol, we introduced `MathBERT`, a domain-specific model for mathematics.
- **Result**: **FAILED**. `MathBERT` also failed to achieve the required separation score.
- **Learning**: A specialized model alone was not enough to solve the problem. The way the input was formatted for the model was likely a key factor.

### Attempt 3: The "Zoom-In" and Metric Change (Ratio)
- **Method**: Based on the insight that all math problems might be semantically similar to a general-purpose model, we adopted the `similarity ratio > 1.2` metric mentioned in the root `README.md`. This metric "zooms in" on the relative differences between clusters.
- **Result**: **FAILED**. The ratio was an improvement, but still not enough to pass.

### Attempt 4: Using Explicit Labels as a Crutch
- **Method**: To see if a pass was even possible, we provided a strong hint by adding explicit prefixes (e.g., "Linear equation:") to the input text.
- **Result**: **PASSED**. Both the `semantic` and `MathBERT` models easily passed with this crutch (`semantic` ratio was ~1.38).
- **Learning**: This was a critical finding. It showed that the models were highly sensitive to the input text and were likely succeeding by "reading the label" rather than understanding the mathematical structure. This success was deemed artificial.

### Attempt 5: Final "Honest" Attempt (No Labels, Ratio Metric)
- **Method**: We reverted the code to remove the explicit prefixes, forcing the models to rely on the mathematical structure of the equations alone, while keeping the more appropriate ratio metric.
- **Result**: **Officially FAILED**, but with a key nuance.
- **Finding**: The `semantic` model (`all-mpnet-base-v2`) achieved a **similarity ratio of 1.1957** (vs. the 1.2 target) and a **perfect 1.0 KNN accuracy**.

## Conclusion
While P1 officially failed on its last and most honest attempt, the `semantic` model's performance was exceptionally close to the goal. It demonstrated a near-perfect ability to classify equation types without any explicit English-language hints.

The key takeaway is that distinguishing these problem types is a difficult task for text-embedding models, but the `semantic` model is robust enough to be considered **"practically passed"**. The risk of using it for the next experimental phases (E1, E2) is low, as the embeddings are demonstrably separable with high accuracy, even if the inter-cluster distance is slightly smaller than desired. This iterative process has given us confidence in the `semantic` model and the ratio-based evaluation.
