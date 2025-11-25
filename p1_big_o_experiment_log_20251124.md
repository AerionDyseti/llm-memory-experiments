# P1 Experiment Log: Big O Complexity - 2025-11-24

## Objective
Following the challenges with the niche mathematical equation experiment, we pivoted to a new domain for our experimental framework. The goal was to find a task that is **objectively verifiable**, **broadly meaningful**, and **sufficiently complex** to avoid a "ceiling effect" where the base model's performance is already perfect. The chosen domain was identifying the Big O time complexity of Python code snippets.

This document records the successful execution of the P1: Embedding Validity test for this new domain.

## Phase 0: Dataset Generation

To support this new experiment, a dataset of code snippets with known Big O complexities was required.

*   **Action**: I created a Python script, `big_o_dataset_generator.py`.
*   **Method**: The script used a template-based approach to generate varied code examples for six distinct complexity classes: `O(1)`, `O(log n)`, `O(n)`, `O(n log n)`, `O(n^2)`, and `O(2^n)`.
*   **Refinement**: Based on user feedback, the script was designed with **multiple distinct templates per class** to ensure the models learn the concept of complexity rather than just memorizing a single code structure.
*   **Output**: The script successfully generated `big_o_dataset.json`, a collection of 300 shuffled code snippets ready for use.

## Phase 1: P1 Embedding Validity Test

With the new dataset, we executed the P1 test to validate that our embedding models could create a well-structured embedding space for the different complexity classes.

*   **Action**: The `prerequisites/p1_embedding_validity.py` script was adapted to use the new `big_o_dataset.json`.
*   **Success Criterion**: We used the `similarity_ratio > 1.2` and `KNN accuracy > 0.70` criteria, which we had previously validated as most appropriate.
*   **Result**: The P1 test **PASSED** on the first attempt.

### Final Pass Results:
The `text` model passed, and the `semantic` model passed with an even stronger result.

*   **'text' Model (`all-MiniLM-L6-v2`)**:
    *   **Similarity Ratio: 1.5871** (Target: > 1.2)
    *   KNN Accuracy: 1.0000 (Target: > 0.70)

*   **'semantic' Model (`all-mpnet-base-v2`)**:
    *   **Similarity Ratio: 1.7110** (Target: > 1.2)
    *   KNN Accuracy: 1.0000 (Target: > 0.70)

## Conclusion & Key Findings

1.  **Successful Pivot**: The Big O complexity domain is a robust and effective choice for our experimental goals, meeting all desired criteria.
2.  **Dataset is Validated**: The template-based generation approach successfully created a dataset with clearly separable classes.
3.  **Embeddings are High-Quality**: General-purpose models like `all-mpnet-base-v2` are highly effective at embedding code and capturing its structural and algorithmic properties. The resulting embedding space is well-structured for retrieval.
4.  **Foundation is Solid**: With P1 passed, we have high confidence in our `semantic` embedding model and can now reliably proceed to the E1 and E2 experiments to test the utility of RAG and fine-tuning.
