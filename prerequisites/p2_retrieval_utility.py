import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
DATASET_PATH = "big_o_dataset.json"
MEMORY_BANK_SIZE = 200 # Examples for retrieval
TEST_SET_SIZE = 50   # Problems for the LLM to solve
RANDOM_SEED = 42

# --- P1 Semantic Embedding Model (passed P1) ---
SEMANTIC_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

# --- Helper Functions ---
def load_dataset(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset

def split_dataset(dataset, memory_size, test_size, seed):
    random.seed(seed)
    random.shuffle(dataset)
    
    if len(dataset) < memory_size + test_size:
        raise ValueError(f"Dataset size ({len(dataset)}) is too small for requested memory_size ({memory_size}) and test_size ({test_size})")

    memory_bank = dataset[:memory_size]
    test_set = dataset[memory_size : memory_size + test_size]
    
    return memory_bank, test_set

def get_embedding(text):
    return semantic_model.encode(text, convert_to_tensor=False)

def get_code_embedding_text(problem):
    return f"{problem['description']}\n```python\n{problem['code']}\n```"

def retrieve_similar_examples(query_embedding, memory_bank_embeddings, memory_bank_problems, num_examples=5):
    similarities = cosine_similarity([query_embedding], memory_bank_embeddings)[0]
    top_indices = similarities.argsort()[-num_examples:][::-1]
    
    retrieved_examples = [memory_bank_problems[i] for i in top_indices]
    return retrieved_examples

# --- Placeholder Agent for LLM Interaction ---
# In a real scenario, this would make an API call to an LLM
# For now, it simulates success/failure or returns ground truth for testing pipeline
def simulate_llm_big_o_analysis(code_snippet, retrieved_context=None, ground_truth=None):
    # This is a placeholder. In a real scenario, this would involve a prompt
    # to an LLM like:
    # prompt = f"""Analyze the following Python code to determine its Big O time complexity.
    # {retrieved_context if retrieved_context else ''}
    # Code:
    # ```python
    # {code_snippet}
    # ```
    # What is the Big O time complexity? Respond only with the complexity (e.g., O(n), O(log n)).
    # """
    # llm_response = call_llm_api(prompt)
    # return parse_llm_response(llm_response)

    # For data pipeline testing, let's just return the ground truth sometimes,
    # or a random one, to simulate an LLM trying to guess.
    
    complexity_classes = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"]
    
    if ground_truth and random.random() < 0.8: # Simulate a somewhat competent LLM
        return ground_truth
    else:
        return random.choice(complexity_classes)

def format_retrieved_examples(examples):
    formatted_str = "\nHere are some examples of code and their Big O complexities:\n"
    for ex in examples:
        formatted_str += f"\nCode ID: {ex['id']}\nComplexity: {ex['complexity_class']}\n```python\n{ex['code']}\n```\n"
    return formatted_str

# --- Main P2 Logic ---
def run_p2_test():
    print("Starting P2: Retrieval Utility Test...")

    # Load and split dataset
    dataset = load_dataset(DATASET_PATH)
    memory_bank, test_set = split_dataset(dataset, MEMORY_BANK_SIZE, TEST_SET_SIZE, RANDOM_SEED)
    
    print(f"Memory bank size: {len(memory_bank)}")
    print(f"Test set size: {len(test_set)}")

    # Pre-compute embeddings for memory bank (semantic model)
    print("Computing embeddings for memory bank...")
    memory_bank_embeddings_text = [get_code_embedding_text(p) for p in memory_bank]
    memory_bank_embeddings = get_embedding(memory_bank_embeddings_text)
    print("Memory bank embeddings computed.")

    results = {
        "no_retrieval": [],
        "random_retrieval": [],
        "semantic_retrieval": []
    }

    # Iterate through the test set
    for i, problem in enumerate(test_set):
        print(f"\nProcessing test problem {i+1}/{len(test_set)} (ID: {problem['id']})")
        code_snippet = problem['code']
        ground_truth = problem['complexity_class']

        # Condition 1: No Retrieval
        llm_prediction_no_retrieval = simulate_llm_big_o_analysis(code_snippet, ground_truth=ground_truth)
        results["no_retrieval"].append(1 if llm_prediction_no_retrieval == ground_truth else 0)
        print(f"  No Retrieval - Predicted: {llm_prediction_no_retrieval}, Ground Truth: {ground_truth} -> {'Correct' if llm_prediction_no_retrieval == ground_truth else 'Incorrect'}")

        # Condition 2: Random Retrieval
        random_examples = random.sample(memory_bank, 5) # Retrieve 5 random examples
        random_context = format_retrieved_examples(random_examples)
        llm_prediction_random_retrieval = simulate_llm_big_o_analysis(code_snippet, retrieved_context=random_context, ground_truth=ground_truth)
        results["random_retrieval"].append(1 if llm_prediction_random_retrieval == ground_truth else 0)
        print(f"  Random Retrieval - Predicted: {llm_prediction_random_retrieval}, Ground Truth: {ground_truth} -> {'Correct' if llm_prediction_random_retrieval == ground_truth else 'Incorrect'}")

        # Condition 3: Semantic Retrieval
        query_embedding_text = get_code_embedding_text(problem)
        query_embedding = get_embedding(query_embedding_text)
        
        semantic_examples = retrieve_similar_examples(query_embedding, memory_bank_embeddings, memory_bank, num_examples=5)
        semantic_context = format_retrieved_examples(semantic_examples)
        llm_prediction_semantic_retrieval = simulate_llm_big_o_analysis(code_snippet, retrieved_context=semantic_context, ground_truth=ground_truth)
        results["semantic_retrieval"].append(1 if llm_prediction_semantic_retrieval == ground_truth else 0)
        print(f"  Semantic Retrieval - Predicted: {llm_prediction_semantic_retrieval}, Ground Truth: {ground_truth} -> {'Correct' if llm_prediction_semantic_retrieval == ground_truth else 'Incorrect'}")

    # Calculate accuracies
    accuracies = {
        condition: np.mean(res) for condition, res in results.items()
    }
    
    print("\n--- P2 Results ---")
    for condition, acc in accuracies.items():
        print(f"{condition.replace('_', ' ').title()} Accuracy: {acc:.4f}")

    # --- P2 Success Criteria (Placeholder for now) ---
    # Need to compare these accuracies statistically.
    # For now, a simple check: is semantic retrieval better than no retrieval?
    
    # Hypothesis from methodology.md: "At least one retrieval method significantly improves performance (p < 0.05)"
    # For simulation, we'll just check if semantic_retrieval accuracy is higher.
    # In a real test, we would run statistical tests (e.g., Wilcoxon signed-rank test as mentioned in old P2).

    p2_passed = accuracies["semantic_retrieval"] > accuracies["no_retrieval"] and \
                accuracies["semantic_retrieval"] > accuracies["random_retrieval"]
    
    if p2_passed:
        print("\nP2: Retrieval Utility Test - PASSED (Simulated)!")
    else:
        print("\nP2: Retrieval Utility Test - FAILED (Simulated).")

    # Save results
    os.makedirs('prerequisites/results', exist_ok=True)
    with open('prerequisites/results/p2_retrieval_analysis.json', 'w') as f:
        json.dump(accuracies, f, indent=4)
    print("Results saved to prerequisites/results/p2_retrieval_analysis.json")


if __name__ == "__main__":
    run_p2_test()
