# LLM Memory Experiments

**Research Question:** Can fine-tuning teach language models to generate better retrieval queries, improving RAG (Retrieval-Augmented Generation) performance?

## Overview

This project investigates whether fine-tuning can create "retrieval intuition" in LLMs - the ability to formulate effective queries when accessing external memory/knowledge bases. We test this through a controlled experiment using code complexity analysis as the domain.

### The Core Hypothesis

Fine-tuning can teach models to generate better retrieval queries ("hunches") even when the base model shows no preference between retrieval strategies.

## Experiment Pipeline

```
Prerequisites → E0: Parameter Optimization → E1: Retrieval Comparison → E2: Fine-tuning
    (P1-P3)         (find best config)        (measure baseline)        (test hypothesis)
```

| Phase | Purpose | Status |
|-------|---------|--------|
| **P1** | Validate embeddings can distinguish problem types | **PASSED** |
| **P2** | Validate retrieval provides utility | In Progress |
| **P3** | Validate fine-tuning preserves capabilities | Pending |
| **E0** | Find optimal retrieval parameters | Pending |
| **E1** | Compare retrieval strategies (temporal vs semantic) | Pending |
| **E2** | Test if fine-tuning improves retrieval usage | Pending |

## Current Progress

### Domain: Big O Complexity Classification

After initial experiments with mathematical equations showed limited separability, we pivoted to **Big O time complexity** - classifying code snippets by their algorithmic complexity (O(1), O(log n), O(n), O(n log n), O(n²), O(2^n)).

This domain is:
- **Objectively verifiable** - complexity classes are well-defined
- **Broadly meaningful** - relevant to real-world programming
- **Sufficiently challenging** - avoids ceiling effects

### P1: Embedding Validity Results

**Objective:** Verify embeddings can distinguish between complexity classes.

| Model | Similarity Ratio | KNN Accuracy | Result |
|-------|------------------|--------------|--------|
| `all-MiniLM-L6-v2` (text) | 1.587 | 100% | **PASS** |
| `all-mpnet-base-v2` (semantic) | **1.711** | 100% | **PASS** |
| MathBERT | 1.087 | 100% | FAIL |

**Target:** Similarity ratio > 1.2, KNN accuracy > 70%

The `semantic` model (`all-mpnet-base-v2`) achieved the best performance with a 1.71 similarity ratio, demonstrating strong cluster separation in the embedding space.

### P2: Retrieval Utility (In Progress)

**Objective:** Verify that retrieval improves model performance.

Preliminary results:
| Condition | Accuracy |
|-----------|----------|
| No retrieval | 86% |
| Random retrieval | 78% |
| Semantic retrieval | 86% |

Analysis ongoing - no significant improvement observed yet.

## Repository Structure

```
llm-memory-experiments/
├── prerequisites/           # Validation experiments (P1-P3)
│   ├── p1_embedding_validity.py
│   ├── p2_retrieval_utility.py
│   ├── results/            # JSON results
│   └── visualizations/     # Cluster plots
├── e0-parameter-optimization/
├── e1-memory-retrieval/
├── e2-fine-tuning/
├── big_o_dataset.json      # Generated code complexity dataset
└── big_o_dataset_generator.py
```

## Key Findings So Far

1. **Domain matters:** Math equations were too semantically similar for general-purpose embeddings. Code complexity provides better separation.

2. **General models work:** Domain-specific models (MathBERT) underperformed general-purpose sentence transformers for our task.

3. **Embeddings validate:** The `all-mpnet-base-v2` model creates well-structured embedding spaces for code classification.

## Next Steps

1. Complete P2 analysis to validate retrieval utility
2. Run P3 to confirm fine-tuning doesn't degrade model capabilities
3. Proceed to E1 to compare temporal vs semantic retrieval strategies
4. Execute E2 to test the core hypothesis

## Technical Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install sentence-transformers scipy numpy pandas matplotlib scikit-learn
```

## Logs & Documentation

- `p1_experiment_log_20251124.md` - Math equation experiment (failed domain)
- `p1_big_o_experiment_log_20251124.md` - Big O experiment (successful domain)
- `prerequisites/methodology.md` - Detailed P1-P3 methodology
- `e*/methodology.md` - Methodology for each experiment phase
