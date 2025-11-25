import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




def create_embeddings(problems, model_name):
    embeddings = []

    if model_name == 'text':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for problem in problems:
            # Use code and description for embedding
            text = f"{problem['description']}\n```python\n{problem['code']}\n```"
            embedding = model.encode(text)
            embeddings.append(embedding)

    elif model_name == 'semantic':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        for problem in problems:
            # Use code and description for embedding
            text = f"{problem['description']}\n```python\n{problem['code']}\n```"
            embedding = model.encode(text)
            embeddings.append(embedding)

    elif model_name == 'mathbert':
        model = SentenceTransformer('math-similarity/Bert-MLM_arXiv-MP-class_zbMath')
        for problem in problems:
            # Use code and description for embedding
            text = f"{problem['description']}\n```python\n{problem['code']}\n```"
            embedding = model.encode(text)
            embeddings.append(embedding)

    return np.array(embeddings)


def analyze_clustering(embeddings, labels, equation_types):
    # Compute similarity matrices
    similarity_matrix = cosine_similarity(embeddings)

    # Intra-class similarity
    intra_similarities = []
    for eq_type in equation_types:
        mask = np.array(labels) == eq_type
        # Ensure mask creates a non-empty selection
        if np.any(mask):
            type_similarities = similarity_matrix[mask][:, mask]
            # Exclude diagonal (self-similarity)
            upper_triangular_indices = np.triu_indices_from(type_similarities, k=1)
            if upper_triangular_indices[0].size > 0: # Check if there are any off-diagonal elements
                intra_similarities.extend(type_similarities[upper_triangular_indices])

    # Inter-class similarity
    inter_similarities = []
    for i, type1 in enumerate(equation_types):
        for type2 in equation_types[i+1:]:
            mask1 = np.array(labels) == type1
            mask2 = np.array(labels) == type2
            # Ensure both masks create non-empty selections
            if np.any(mask1) and np.any(mask2):
                inter_similarities.extend(similarity_matrix[mask1][:, mask2].flatten())
    
    # Handle cases where there might be no similarities if some types are missing or only one instance
    intra_mean = np.mean(intra_similarities) if intra_similarities else 0
    intra_std = np.std(intra_similarities) if intra_similarities else 0
    inter_mean = np.mean(inter_similarities) if inter_similarities else 0
    inter_std = np.std(inter_similarities) if inter_similarities else 0


    separation = intra_mean - inter_mean
    # Calculate similarity ratio as per README.md suggestion, handle division by zero
    similarity_ratio = intra_mean / inter_mean if inter_mean != 0 else float('inf')

    return {
        'intra_mean': intra_mean,
        'intra_std': intra_std,
        'inter_mean': inter_mean,
        'inter_std': inter_std,
        'separation': separation,
        'similarity_ratio': similarity_ratio
    }

def visualize_embeddings(embeddings, labels, equation_types, filename='embedding_clusters.png'):
    # Check if there are enough samples for t-SNE
    if embeddings.shape[0] < 2:
        print("Not enough samples for t-SNE visualization.")
        return

    # t-SNE projection
    # Perplexity must be less than the number of samples.
    # If there are fewer than 50 samples, set perplexity to N-1
    perplexity_val = min(30, embeddings.shape[0] - 1)
    if perplexity_val <= 0:
        print("Not enough samples for t-SNE with a valid perplexity.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    colors = {'linear': 'red', 'quadratic': 'blue',
              'exponential': 'green', 'trigonometric': 'purple'}

    plt.figure(figsize=(10, 8))
    for eq_type in equation_types:
        mask = np.array(labels) == eq_type
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors.get(eq_type, 'gray'), label=eq_type, alpha=0.6) # Use .get with default for safety
    plt.legend()
    plt.title('t-SNE Visualization of Equation Embeddings')
    plt.savefig(filename)
    plt.close() # Close plot to free memory

def evaluate_knn_classifier(embeddings, labels, equation_types, n_neighbors=5):
    if len(labels) < n_neighbors + 1: # Need enough samples for KNN and train/test split
        print(f"Not enough samples ({len(labels)}) for KNN with n_neighbors={n_neighbors}. Skipping KNN evaluation.")
        return 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )

    if len(np.unique(y_train)) < 2: # Ensure at least two classes in training data
        print("Not enough classes in training data for KNN. Skipping KNN evaluation.")
        return 0.0
    
    # Ensure n_neighbors is not greater than the number of samples in any class in training data
    # This might require adjusting n_neighbors or the splitting strategy for very small datasets
    for eq_type in equation_types:
        if sum(1 for y in y_train if y == eq_type) < n_neighbors:
            print(f"Warning: Class '{eq_type}' in training data has less than {n_neighbors} samples. KNN might struggle or fail for this class.")
            # Adjust n_neighbors if necessary, or proceed with caution.
            # For simplicity, we'll proceed, but in a robust system, you might reduce n_neighbors.
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    return accuracy

if __name__ == "__main__":
    # Ensure the directory for results and visualizations exists
    import os
    os.makedirs('prerequisites/results', exist_ok=True)
    os.makedirs('prerequisites/visualizations', exist_ok=True)

    print("Starting P1: Big O Embedding Validity Test...")
    
    # Load the Big O dataset
    with open('big_o_dataset.json', 'r') as f:
        problems = json.load(f)

    # Define the new complexity classes
    complexity_classes = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"]
    labels = [p['complexity_class'] for p in problems]

    embedding_results = {}

    for emb_model_name in ['text', 'semantic', 'mathbert']: # Hybrid model removed
        print(f"\nCreating embeddings for model: {emb_model_name}...")
        embeddings = create_embeddings(problems, emb_model_name)
        
        if embeddings.size == 0:
            print(f"No embeddings generated for {emb_model_name}. Skipping analysis.")
            continue

        print(f"Analyzing clustering for model: {emb_model_name}...")
        # Pass complexity_classes to analyze_clustering
        clustering_analysis = analyze_clustering(embeddings, labels, complexity_classes)
        print(f"  Clustering Analysis ({emb_model_name}):")
        for key, value in clustering_analysis.items():
            print(f"    {key}: {value:.4f}")
        
        # 2. KNN Classification Accuracy
        # Pass complexity_classes to evaluate_knn_classifier
        knn_accuracy = evaluate_knn_classifier(embeddings, labels, complexity_classes)
        print(f"  KNN Classification Accuracy ({emb_model_name}): {knn_accuracy:.4f}")

        # 3. Visualization
        # Update colors for new complexity classes
        colors = {
            "O(1)": 'red',
            "O(log n)": 'green',
            "O(n)": 'blue',
            "O(n log n)": 'orange',
            "O(n^2)": 'purple',
            "O(2^n)": 'brown'
        }
        
        plt.figure(figsize=(10, 8))
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
        embeddings_2d = tsne.fit_transform(embeddings)

        for comp_class in complexity_classes:
            mask = np.array(labels) == comp_class
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=colors.get(comp_class, 'gray'), label=comp_class, alpha=0.6)
        plt.legend()
        plt.title(f't-SNE Visualization of Big O Embeddings ({emb_model_name})')
        plt.savefig(f'prerequisites/visualizations/p1_big_o_{emb_model_name}_embedding_clusters.png')
        plt.close()
        print(f"  t-SNE plot saved to prerequisites/visualizations/p1_big_o_{emb_model_name}_embedding_clusters.png")

        embedding_results[emb_model_name] = {
            'clustering_analysis': clustering_analysis,
            'knn_accuracy': knn_accuracy
        }
    
    # Determine overall success
    overall_pass = False
    for emb_model_name, res in embedding_results.items():
        sim_ratio = res['clustering_analysis']['similarity_ratio']
        knn_acc = res['knn_accuracy']
        
        if sim_ratio > 1.2 and knn_acc > 0.70:
            print(f"\nSUCCESS: Embedding model '{emb_model_name}' passed P1 criteria!")
            print(f"  - Similarity Ratio: {sim_ratio:.4f} (> 1.2)")
            print(f"  - KNN Accuracy: {knn_acc:.4f} (> 0.70)")
            overall_pass = True
            break # One model passing is enough for the overall P1 pass criteria

    if overall_pass:
        print("\nP1: Big O Embedding Validity Test - PASSED!")
    else:
        print("\nP1: Big O Embedding Validity Test - FAILED. Review methodology and consider failure protocols.")

    # Save results to JSON
    def convert_numpy_to_python(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy_to_python(elem) for elem in obj]
        return obj

    python_serializable_results = convert_numpy_to_python(embedding_results)

    with open('prerequisites/results/p1_big_o_embedding_analysis.json', 'w') as f:
        json.dump(python_serializable_results, f, indent=4)
    print("Results saved to prerequisites/results/p1_big_o_embedding_analysis.json")
