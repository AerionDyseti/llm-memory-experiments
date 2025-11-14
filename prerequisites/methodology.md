# Prerequisites: Critical Validation Experiments

## Overview
These three experiments must pass before proceeding with the main experimental chain. They validate fundamental assumptions that, if false, would invalidate all subsequent experiments.

---

## P1: Embedding Validity Test

### Purpose
Verify that our embedding approach can meaningfully distinguish between different types of mathematical problems.

### Hypothesis
- **H₁**: Embeddings of similar equation types will cluster together with intra-class similarity > inter-class similarity + 0.3
- **H₀**: Embeddings show no meaningful clustering (similarity is random)

### Methodology

#### 1. Data Generation
```python
# Generate 100 test problems
problems_per_type = 25
equation_types = ['linear', 'quadratic', 'exponential', 'trigonometric']

def generate_problems():
    problems = []
    for eq_type in equation_types:
        for i in range(problems_per_type):
            if eq_type == 'linear':
                a = random.uniform(-10, 10)
                b = random.uniform(-20, 20)
                equation = f"{a}*x + {b}"
            elif eq_type == 'quadratic':
                a = random.uniform(-5, 5)
                b = random.uniform(-5, 5)
                c = random.uniform(-5, 5)
                equation = f"{a}*x**2 + {b}*x + {c}"
            elif eq_type == 'exponential':
                a = random.uniform(1, 5)
                b = random.uniform(1.1, 3)
                equation = f"{a} * {b}**x"
            elif eq_type == 'trigonometric':
                a = random.uniform(1, 10)
                b = random.uniform(0.5, 2)
                c = random.uniform(-5, 5)
                equation = f"{a} * sin({b}*x) + {c}"

            # Generate data points
            x_values = np.linspace(-10, 10, 10)
            y_values = [eval(equation.replace('x', str(x))) for x in x_values]
            y_noisy = y_values + np.random.normal(0, 0.1, len(y_values))

            problems.append({
                'id': f"{eq_type}_{i}",
                'type': eq_type,
                'equation': equation,
                'data_points': list(zip(x_values, y_noisy))
            })
    return problems
```

#### 2. Embedding Generation
```python
# Test three embedding approaches
embedding_models = {
    'text': 'sentence-transformers/all-MiniLM-L6-v2',
    'semantic': 'sentence-transformers/all-mpnet-base-v2',
    'hybrid': 'custom'  # Combination of text + numerical features
}

def create_embeddings(problems, model_name):
    embeddings = []

    if model_name in ['text', 'semantic']:
        model = SentenceTransformer(embedding_models[model_name])
        for problem in problems:
            # Format as text
            text = f"Data points: {problem['data_points']}"
            embedding = model.encode(text)
            embeddings.append(embedding)

    elif model_name == 'hybrid':
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        for problem in problems:
            # Text embedding
            text = f"Data points: {problem['data_points']}"
            text_emb = text_model.encode(text)

            # Numerical features
            x_vals = [p[0] for p in problem['data_points']]
            y_vals = [p[1] for p in problem['data_points']]
            features = [
                np.mean(y_vals),
                np.std(y_vals),
                np.corrcoef(x_vals, y_vals)[0,1],
                (max(y_vals) - min(y_vals)) / (max(x_vals) - min(x_vals))
            ]

            # Combine
            embedding = np.concatenate([text_emb * 0.7, features * 0.3])
            embeddings.append(embedding)

    return np.array(embeddings)
```

#### 3. Clustering Analysis
```python
def analyze_clustering(embeddings, labels):
    # Compute similarity matrices
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Intra-class similarity
    intra_similarities = []
    for eq_type in equation_types:
        mask = labels == eq_type
        type_similarities = similarity_matrix[mask][:, mask]
        # Exclude diagonal
        intra_similarities.extend(type_similarities[np.triu_indices_from(type_similarities, k=1)])

    # Inter-class similarity
    inter_similarities = []
    for i, type1 in enumerate(equation_types):
        for type2 in equation_types[i+1:]:
            mask1 = labels == type1
            mask2 = labels == type2
            inter_similarities.extend(similarity_matrix[mask1][:, mask2].flatten())

    return {
        'intra_mean': np.mean(intra_similarities),
        'intra_std': np.std(intra_similarities),
        'inter_mean': np.mean(inter_similarities),
        'inter_std': np.std(inter_similarities),
        'separation': np.mean(intra_similarities) - np.mean(inter_similarities)
    }
```

#### 4. Visualization
```python
def visualize_embeddings(embeddings, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    colors = {'linear': 'red', 'quadratic': 'blue',
              'exponential': 'green', 'trigonometric': 'purple'}

    plt.figure(figsize=(10, 8))
    for eq_type in equation_types:
        mask = labels == eq_type
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors[eq_type], label=eq_type, alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of Equation Embeddings')
    plt.savefig('embedding_clusters.png')
```

### Success Criteria
- **Primary**: Separation score > 0.3 (intra-class - inter-class similarity)
- **Secondary**: KNN classification accuracy > 70% with k=5
- **Visual**: Clear clustering in t-SNE plot

### Failure Protocol
If test fails:
1. Try mathematical-specific embeddings (MathBERT)
2. Increase weight of numerical features
3. Consider symbolic representation approaches
4. If all fail: STOP experiments, fundamental redesign needed

### Time Estimate
- Setup: 30 minutes
- Data generation: 30 minutes
- Embedding computation: 1 hour
- Analysis: 1 hour
- Total: 3 hours

---

## P2: Retrieval Utility Test

### Purpose
Verify that access to solved examples improves problem-solving performance.

### Hypothesis
- **H₁**: At least one retrieval method significantly improves performance (p < 0.05)
- **H₀**: No retrieval method helps (all p ≥ 0.05)

### Methodology

#### 1. Create Mini Memory Bank
```python
def create_memory_bank():
    # Generate 100 solved problems
    memory_problems = generate_problems(n=100)
    memory_bank = []

    for problem in memory_problems:
        # Simulate solving (for testing, use known equation)
        memory_bank.append({
            'problem_id': problem['id'],
            'data_points': problem['data_points'],
            'solution': problem['equation'],
            'attempts': random.randint(3, 10),
            'strategy': random.choice([
                'Noticed linear pattern',
                'Detected quadratic growth',
                'Found exponential scaling',
                'Identified periodic behavior'
            ])
        })

    return memory_bank
```

#### 2. Test Conditions
```python
def run_conditions(test_problems, memory_bank):
    results = {
        'no_retrieval': [],
        'random_retrieval': [],
        'recent_retrieval': []
    }

    for problem in test_problems:
        # Condition A: No retrieval
        attempts_none = solve_without_retrieval(problem)
        results['no_retrieval'].append(attempts_none)

        # Condition B: Random 5 retrievals
        random_memories = random.sample(memory_bank, 5)
        attempts_random = solve_with_retrieval(problem, random_memories)
        results['random_retrieval'].append(attempts_random)

        # Condition C: Recent 5 retrievals
        recent_memories = memory_bank[-5:]
        attempts_recent = solve_with_retrieval(problem, recent_memories)
        results['recent_retrieval'].append(attempts_recent)

    return results
```

#### 3. Agent Implementation
```python
def solve_without_retrieval(problem, max_attempts=20):
    prompt = f"""
    Given these data points: {problem['data_points']}
    Find the underlying equation.
    You have {max_attempts} attempts.
    """

    # Run agent
    attempts = run_agent(prompt, problem['equation'], max_attempts)
    return attempts

def solve_with_retrieval(problem, retrieved_memories, max_attempts=20):
    memory_text = format_memories(retrieved_memories)
    prompt = f"""
    Given these data points: {problem['data_points']}

    Here are some similar solved problems:
    {memory_text}

    Find the underlying equation.
    You have {max_attempts} attempts.
    """

    attempts = run_agent(prompt, problem['equation'], max_attempts)
    return attempts
```

#### 4. Statistical Analysis
```python
def analyze_results(results):
    from scipy import stats

    # Pairwise comparisons
    comparisons = []

    # No retrieval vs Random
    stat1, p1 = stats.wilcoxon(results['no_retrieval'],
                                results['random_retrieval'])
    comparisons.append(('none_vs_random', p1))

    # No retrieval vs Recent
    stat2, p2 = stats.wilcoxon(results['no_retrieval'],
                                results['recent_retrieval'])
    comparisons.append(('none_vs_recent', p2))

    # Effect sizes
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        dx = (nx-1) * np.var(x, ddof=1)
        dy = (ny-1) * np.var(y, ddof=1)
        pooled_var = (dx + dy) / (nx + ny - 2)
        return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)

    effect_sizes = {
        'random': cohen_d(results['no_retrieval'], results['random_retrieval']),
        'recent': cohen_d(results['no_retrieval'], results['recent_retrieval'])
    }

    return comparisons, effect_sizes
```

### Success Criteria
- **Primary**: At least one p-value < 0.05
- **Secondary**: Effect size (Cohen's d) > 0.3 for significant comparisons
- **Practical**: Mean improvement > 2 attempts

### Failure Protocol
If no retrieval helps:
1. Check if problems are too easy (ceiling effect)
2. Check if problems are too hard (floor effect)
3. Try different retrieval presentation formats
4. Investigate agent's ability to use examples
5. If still fails: STOP - fundamental issue with approach

### Time Estimate
- Memory bank creation: 1 hour
- Test setup: 30 minutes
- Running conditions (30 problems × 3 conditions × 3 runs): 2 hours
- Analysis: 30 minutes
- Total: 4 hours

---

## P3: Fine-tuning Preservation Test

### Purpose
Verify that LoRA fine-tuning preserves the model's core capabilities while adding new ones.

### Hypothesis
- **H₁**: Fine-tuned model retains >80% of base model performance on standard benchmarks
- **H₀**: Fine-tuning causes significant capability degradation

### Methodology

#### 1. Create Aggressive Training Set
```python
def create_training_data():
    # Small but intensive training set
    training_problems = generate_problems(n=50)
    training_data = []

    for problem in training_problems:
        training_data.append({
            'instruction': 'Analyze the data and find the pattern',
            'input': f"Data: {problem['data_points']}",
            'output': f"The equation is {problem['equation']}"
        })

    return training_data
```

#### 2. Aggressive Fine-tuning
```python
def aggressive_finetune(training_data):
    config = {
        'model': 'phi-3-mini-4k-instruct',
        'r': 16,  # Higher rank for more aggressive
        'alpha': 32,
        'dropout': 0.05,  # Lower dropout
        'learning_rate': 1e-4,  # Higher LR
        'epochs': 5,  # More epochs
        'batch_size': 4
    }

    # Fine-tune with mlx
    adapter = train_lora(training_data, config)
    return adapter
```

#### 3. Capability Benchmarks
```python
def test_capabilities(model, is_finetuned=False):
    benchmarks = {
        'instruction_following': [],
        'math_reasoning': [],
        'format_compliance': [],
        'general_knowledge': []
    }

    # Test 1: Instruction Following
    instruction_prompts = [
        "List three prime numbers",
        "Write a haiku about mathematics",
        "Explain addition in one sentence",
        # ... 20 total
    ]

    # Test 2: Math Reasoning
    math_problems = [
        "What is 15 + 27?",
        "If x = 5, what is 2x + 3?",
        "Solve: 3y = 12",
        # ... 20 total
    ]

    # Test 3: Format Compliance
    format_prompts = [
        "Output JSON: {'name': 'test', 'value': 123}",
        "Create a markdown table with 2 rows",
        "Write exactly 5 words",
        # ... 20 total
    ]

    # Test 4: General Knowledge
    knowledge_prompts = [
        "What is the capital of France?",
        "Name a programming language",
        "What year did Python release?",
        # ... 20 total
    ]

    # Run all benchmarks and score
    for category, prompts in [...]:
        scores = evaluate_responses(model, prompts)
        benchmarks[category] = scores

    return benchmarks
```

#### 4. Performance Comparison
```python
def compare_performance(base_scores, finetuned_scores):
    retention_rates = {}

    for category in base_scores.keys():
        base_mean = np.mean(base_scores[category])
        tuned_mean = np.mean(finetuned_scores[category])
        retention = (tuned_mean / base_mean) * 100
        retention_rates[category] = retention

    overall_retention = np.mean(list(retention_rates.values()))

    return {
        'category_retention': retention_rates,
        'overall_retention': overall_retention,
        'pass': overall_retention >= 80
    }
```

### Success Criteria
- **Primary**: Overall capability retention ≥ 80%
- **Secondary**: No single category drops below 60%
- **Critical**: Model still follows instruction format

### Failure Protocol
If capabilities degrade:
1. Reduce LoRA rank (r=4 or r=2)
2. Lower learning rate (5e-5 or 1e-5)
3. Add regularization (increase dropout)
4. Reduce training epochs
5. Try different LoRA target modules
6. If still fails: Consider alternative fine-tuning methods

### Time Estimate
- Training data prep: 30 minutes
- Fine-tuning: 1 hour
- Benchmark testing: 1 hour
- Analysis: 30 minutes
- Total: 3 hours

---

## Integration Points

### Data Flow
```
P1 → Validates embedding approach → E0 uses best embedding model
P2 → Validates retrieval helps → E1 tests retrieval strategies
P3 → Validates fine-tuning safety → E2 uses safe parameters
```

### Shared Resources
- Problem generator function (used by all)
- Agent runner infrastructure (P2, E1, E2)
- Statistical analysis tools (all experiments)

### Output Files
```
prerequisites/
├── results/
│   ├── p1_embedding_analysis.json
│   ├── p2_retrieval_comparison.json
│   └── p3_capability_retention.json
├── visualizations/
│   ├── embedding_clusters.png
│   ├── retrieval_performance.png
│   └── capability_comparison.png
└── logs/
    └── prerequisite_tests.log
```

## Risk Matrix

| Risk | Impact | Mitigation | Decision Point |
|------|---------|------------|----------------|
| Embeddings don't cluster | Fatal - Stop all | Try domain-specific models | After P1 |
| Retrieval doesn't help | Fatal - Stop all | Check problem difficulty | After P2 |
| Fine-tuning breaks model | Major - Modify E2 | Adjust parameters | After P3 |
| Tests take too long | Minor - Delay | Run in parallel where possible | During execution |

## Computational Requirements

- **Hardware**: M2 Pro (CPU for P1/P2, Neural Engine for P3)
- **Memory**: 4GB RAM minimum
- **Storage**: 2GB for models and data
- **Time**: 10 hours total (can parallelize some parts)