# E0: Parameter Optimization for Semantic Retrieval

## Overview
This experiment determines optimal parameters for semantic retrieval before running the main E1 experiment. We perform a grid search over embedding configurations and retrieval parameters.

## Research Question
What combination of embedding model, feature weights, and retrieval count maximizes the relevance of retrieved examples for mathematical problem-solving?

## Hypotheses
- **H₁**: Hybrid embeddings (text + numerical features) will outperform text-only embeddings
- **H₂**: An optimal retrieval count exists that balances information vs. noise (expected: 5-7)
- **H₃**: Mathematical features will require higher weight (>0.3) for equation problems

---

## Methodology

### 1. Parameter Space Definition

```python
PARAMETER_GRID = {
    'embedding_models': [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'angle-bert-base'  # If available
    ],
    'feature_weights': [
        (1.0, 0.0),  # Text only
        (0.7, 0.3),  # Balanced toward text
        (0.5, 0.5),  # Equal weight
        (0.3, 0.7),  # Balanced toward features
        (0.0, 1.0),  # Features only
    ],
    'retrieval_counts': [3, 5, 7, 10],
    'similarity_metrics': ['cosine', 'euclidean']
}

# Total combinations: 3 × 5 × 4 × 2 = 120 configurations
```

### 2. Data Preparation

#### 2.1 Create Validation Set
```python
def create_validation_data():
    """
    Create 200 problems from memory set for parameter tuning.
    Keep separate from main experiment data.
    """
    np.random.seed(42)  # Reproducibility

    validation_problems = []
    problem_types = ['linear', 'quadratic', 'exponential', 'trigonometric']

    for ptype in problem_types:
        for i in range(50):  # 50 per type
            problem = generate_problem(ptype, seed=hash(f"{ptype}_{i}"))
            validation_problems.append(problem)

    # Create memory bank from first 150
    memory_bank = validation_problems[:150]

    # Use last 50 as test queries
    query_problems = validation_problems[150:]

    return memory_bank, query_problems
```

#### 2.2 Feature Extraction
```python
def extract_numerical_features(data_points):
    """
    Extract mathematical features from data points.
    """
    x_vals = np.array([p[0] for p in data_points])
    y_vals = np.array([p[1] for p in data_points])

    # Basic statistics
    features = {
        'y_mean': np.mean(y_vals),
        'y_std': np.std(y_vals),
        'y_range': np.ptp(y_vals),
        'y_skew': scipy.stats.skew(y_vals),
        'y_kurtosis': scipy.stats.kurtosis(y_vals)
    }

    # Relationship features
    features['correlation'] = np.corrcoef(x_vals, y_vals)[0, 1]

    # Derivative approximation (for detecting growth patterns)
    dy_dx = np.diff(y_vals) / np.diff(x_vals)
    features['mean_derivative'] = np.mean(dy_dx)
    features['derivative_variance'] = np.var(dy_dx)

    # Second derivative (for detecting curvature)
    d2y_dx2 = np.diff(dy_dx) / np.diff(x_vals[:-1])
    features['mean_curvature'] = np.mean(d2y_dx2)

    # Periodicity check (for trig functions)
    fft = np.fft.fft(y_vals)
    features['spectral_peak'] = np.max(np.abs(fft[1:len(fft)//2]))

    return np.array(list(features.values()))
```

### 3. Grid Search Implementation

#### 3.1 Main Search Loop
```python
def grid_search(memory_bank, query_problems, parameter_grid):
    """
    Exhaustive search over parameter space.
    """
    results = []

    for model_name in parameter_grid['embedding_models']:
        print(f"Testing model: {model_name}")
        model = load_embedding_model(model_name)

        for text_weight, feature_weight in parameter_grid['feature_weights']:
            for k in parameter_grid['retrieval_counts']:
                for metric in parameter_grid['similarity_metrics']:

                    config = {
                        'model': model_name,
                        'text_weight': text_weight,
                        'feature_weight': feature_weight,
                        'k': k,
                        'metric': metric
                    }

                    # Evaluate configuration
                    performance = evaluate_configuration(
                        model, memory_bank, query_problems, config
                    )

                    results.append({
                        **config,
                        **performance
                    })

                    # Save intermediate results
                    save_checkpoint(results)

    return results
```

#### 3.2 Configuration Evaluation
```python
def evaluate_configuration(model, memory_bank, query_problems, config):
    """
    Evaluate a single parameter configuration.
    """
    # Create embeddings for memory bank
    memory_embeddings = create_hybrid_embeddings(
        memory_bank, model, config['text_weight'], config['feature_weight']
    )

    metrics = {
        'relevance_scores': [],
        'type_match_rate': [],
        'solution_similarity': [],
        'diversity_scores': [],
        'retrieval_time': []
    }

    for query in query_problems:
        start_time = time.time()

        # Create query embedding
        query_embedding = create_hybrid_embedding(
            query, model, config['text_weight'], config['feature_weight']
        )

        # Retrieve k most similar
        retrieved_indices = retrieve_similar(
            query_embedding, memory_embeddings,
            k=config['k'], metric=config['metric']
        )

        retrieval_time = time.time() - start_time

        # Evaluate retrieval quality
        retrieved_problems = [memory_bank[i] for i in retrieved_indices]

        # Metric 1: Type match rate
        type_matches = sum(1 for p in retrieved_problems
                          if p['type'] == query['type'])
        metrics['type_match_rate'].append(type_matches / config['k'])

        # Metric 2: Solution similarity (for oracle evaluation)
        solution_similarities = [
            equation_similarity(p['equation'], query['equation'])
            for p in retrieved_problems
        ]
        metrics['solution_similarity'].append(np.mean(solution_similarities))

        # Metric 3: Diversity (want some diversity, not all identical)
        diversity = compute_diversity(retrieved_problems)
        metrics['diversity_scores'].append(diversity)

        # Metric 4: Relevance (combination of above)
        relevance = compute_relevance_score(
            type_matches, solution_similarities, diversity
        )
        metrics['relevance_scores'].append(relevance)

        metrics['retrieval_time'].append(retrieval_time)

    return {
        'mean_relevance': np.mean(metrics['relevance_scores']),
        'std_relevance': np.std(metrics['relevance_scores']),
        'mean_type_match': np.mean(metrics['type_match_rate']),
        'mean_diversity': np.mean(metrics['diversity_scores']),
        'mean_retrieval_time': np.mean(metrics['retrieval_time']),
        'efficiency_score': np.mean(metrics['relevance_scores']) /
                           np.mean(metrics['retrieval_time'])
    }
```

### 4. Optimization Metrics

#### 4.1 Relevance Scoring
```python
def compute_relevance_score(type_matches, solution_similarities, diversity):
    """
    Combine multiple factors into single relevance score.
    """
    # Weights determined by domain knowledge
    weights = {
        'type_match': 0.4,    # Same equation type is important
        'solution_sim': 0.4,  # Similar solutions help
        'diversity': 0.2      # Some variety is good
    }

    score = (weights['type_match'] * (type_matches / k) +
             weights['solution_sim'] * np.mean(solution_similarities) +
             weights['diversity'] * diversity)

    return score
```

#### 4.2 Diversity Measurement
```python
def compute_diversity(retrieved_problems):
    """
    Measure diversity of retrieved set.
    """
    # Type diversity
    types = [p['type'] for p in retrieved_problems]
    type_diversity = len(set(types)) / len(types)

    # Solution diversity (avoid all identical)
    solutions = [p['equation'] for p in retrieved_problems]
    unique_patterns = len(set([classify_pattern(s) for s in solutions]))
    solution_diversity = unique_patterns / len(solutions)

    return (type_diversity + solution_diversity) / 2
```

### 5. Analysis and Selection

#### 5.1 Performance Analysis
```python
def analyze_results(results_df):
    """
    Analyze grid search results to find optimal configuration.
    """
    # Find Pareto frontier (relevance vs efficiency)
    pareto_configs = find_pareto_frontier(
        results_df[['mean_relevance', 'efficiency_score']].values
    )

    # Analyze factor importance
    factor_importance = {
        'model': analyze_factor(results_df, 'model'),
        'text_weight': analyze_factor(results_df, 'text_weight'),
        'k': analyze_factor(results_df, 'k'),
        'metric': analyze_factor(results_df, 'metric')
    }

    # Find top configurations
    top_by_relevance = results_df.nlargest(5, 'mean_relevance')
    top_by_efficiency = results_df.nlargest(5, 'efficiency_score')

    # Statistical tests
    best_config = results_df.iloc[results_df['mean_relevance'].idxmax()]

    # Bootstrap confidence intervals
    ci_relevance = bootstrap_ci(best_config['relevance_scores'])

    return {
        'best_config': best_config,
        'pareto_frontier': pareto_configs,
        'factor_importance': factor_importance,
        'confidence_interval': ci_relevance,
        'top_5_relevance': top_by_relevance,
        'top_5_efficiency': top_by_efficiency
    }
```

#### 5.2 Visualization
```python
def create_visualizations(results_df):
    """
    Create comprehensive visualizations of results.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Heatmap: text_weight vs k
    pivot = results_df.pivot_table(
        values='mean_relevance',
        index='text_weight',
        columns='k',
        aggfunc='mean'
    )
    sns.heatmap(pivot, ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title('Relevance: Text Weight vs K')

    # 2. Model comparison
    sns.boxplot(data=results_df, x='model', y='mean_relevance', ax=axes[0,1])
    axes[0,1].set_title('Model Performance')
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Efficiency vs Relevance scatter
    scatter = axes[0,2].scatter(
        results_df['mean_relevance'],
        results_df['efficiency_score'],
        c=results_df['k'],
        cmap='coolwarm'
    )
    axes[0,2].set_xlabel('Relevance')
    axes[0,2].set_ylabel('Efficiency')
    axes[0,2].set_title('Pareto Frontier')
    plt.colorbar(scatter, ax=axes[0,2], label='K')

    # 4. K value analysis
    sns.lineplot(data=results_df, x='k', y='mean_relevance',
                 hue='model', ax=axes[1,0])
    axes[1,0].set_title('Retrieval Count Impact')

    # 5. Feature weight impact
    feature_only = results_df[results_df['text_weight'] == 0]
    text_only = results_df[results_df['text_weight'] == 1]
    hybrid = results_df[(results_df['text_weight'] > 0) &
                       (results_df['text_weight'] < 1)]

    comparison = pd.DataFrame({
        'Features Only': feature_only['mean_relevance'].mean(),
        'Text Only': text_only['mean_relevance'].mean(),
        'Hybrid': hybrid['mean_relevance'].mean()
    }, index=[0])

    comparison.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Embedding Type Comparison')
    axes[1,1].set_ylabel('Mean Relevance')

    # 6. Time vs Performance
    axes[1,2].scatter(results_df['mean_retrieval_time'],
                     results_df['mean_relevance'])
    axes[1,2].set_xlabel('Retrieval Time (s)')
    axes[1,2].set_ylabel('Relevance')
    axes[1,2].set_title('Time-Performance Trade-off')

    plt.tight_layout()
    plt.savefig('e0_optimization_results.png', dpi=300)
```

### 6. Configuration Selection

```python
def select_final_configuration(analysis_results, constraints=None):
    """
    Select final configuration based on analysis and constraints.
    """
    if constraints is None:
        constraints = {
            'max_retrieval_time': 0.1,  # 100ms max
            'min_relevance': 0.7,        # 70% relevance minimum
            'preferred_k': [5, 7]         # Prefer moderate K
        }

    candidates = analysis_results['pareto_frontier']

    # Apply constraints
    valid = candidates[
        (candidates['mean_retrieval_time'] <= constraints['max_retrieval_time']) &
        (candidates['mean_relevance'] >= constraints['min_relevance']) &
        (candidates['k'].isin(constraints['preferred_k']))
    ]

    if len(valid) == 0:
        print("No configuration meets all constraints. Relaxing...")
        # Relax constraints and try again
        constraints['min_relevance'] *= 0.9
        return select_final_configuration(analysis_results, constraints)

    # Select best from valid candidates
    final = valid.iloc[valid['mean_relevance'].idxmax()]

    # Create config file for E1
    config_for_e1 = {
        'embedding_model': final['model'],
        'text_weight': final['text_weight'],
        'feature_weight': final['feature_weight'],
        'retrieval_count': final['k'],
        'similarity_metric': final['metric'],
        'expected_relevance': final['mean_relevance'],
        'confidence_interval': analysis_results['confidence_interval']
    }

    # Save configuration
    with open('e1_config.json', 'w') as f:
        json.dump(config_for_e1, f, indent=2)

    return config_for_e1
```

---

## Output Specifications

### Files Generated
```
e0-parameter-optimization/
├── results/
│   ├── grid_search_results.csv
│   ├── analysis_summary.json
│   ├── e1_config.json  # Final configuration for E1
│   └── pareto_frontier.csv
├── visualizations/
│   ├── optimization_results.png
│   ├── relevance_heatmap.png
│   └── efficiency_frontier.png
├── checkpoints/
│   └── grid_search_checkpoint_*.pkl
└── logs/
    └── optimization.log
```

### Key Metrics Reported
1. **Optimal Configuration**
   - Model name
   - Weight balance (text vs features)
   - Retrieval count (k)
   - Expected relevance score

2. **Performance Bounds**
   - Best relevance achieved
   - Confidence intervals
   - Efficiency scores
   - Time requirements

3. **Insights**
   - Most important factors
   - Trade-off curves
   - Unexpected findings

---

## Execution Plan

### Phase 1: Setup (1 hour)
1. Load embedding models
2. Generate validation data
3. Initialize logging

### Phase 2: Grid Search (6 hours)
1. Run 120 configurations
2. ~3 minutes per configuration
3. Checkpoint every 10 configurations

### Phase 3: Analysis (1 hour)
1. Statistical analysis
2. Generate visualizations
3. Select final configuration
4. Create E1 config file

### Total Time: 8 hours

---

## Risk Mitigation

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Grid search too slow | Parallelize evaluations | Reduce grid granularity |
| No clear optimum | Use ensemble of top-3 | Default to balanced config |
| Memory issues | Process in batches | Reduce validation set size |
| Model unavailable | Skip that model | Use available models only |

---

## Success Criteria

1. **Primary**: Find configuration with relevance > 0.75
2. **Secondary**: Retrieval time < 100ms
3. **Tertiary**: Clear Pareto frontier identified

---

## Dependencies from Prerequisites

- P1 must confirm embeddings work (similarity ratio > 1.2)
- Uses validation approach from P2
- Inherits safe parameters from P3