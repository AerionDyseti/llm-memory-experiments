# E2: Fine-tuning for Intuition (Main Experiment)

## Overview
This is the main experiment testing whether fine-tuning on different data formats (distilled summaries vs raw logs) can improve an agent's ability to use RAG memory effectively. The specific hypothesis tested depends on E1 results.

## Research Questions
1. Can fine-tuning improve how models generate retrieval queries ("hunches")?
2. Which training data format provides better cost-efficiency trade-offs?
3. Does the improvement mechanism vary based on base model capabilities?

## Adaptive Hypotheses (Based on E1)
- **Amplification**: If E1 shows semantic >> temporal, test if fine-tuning amplifies advantage
- **Awakening**: If E1 shows semantic ≈ temporal, test if fine-tuning creates advantage
- **Correction**: If E1 shows semantic < temporal, test if fine-tuning fixes bias
- **Stabilization**: If E1 shows high variance, test if fine-tuning reduces variance

---

## Experimental Design

### Adaptive Design Based on E1 Results

```python
def select_e2_design(e1_results):
    """
    Select E2 experimental design based on E1 findings.
    """

    hypothesis = e1_results['hypothesis']

    if hypothesis == 'amplification':
        # E1 showed strong semantic advantage
        return {
            'conditions': ['base_semantic', 'finetuned_s', 'finetuned_r'],
            'focus': 'enhancement',
            'training_emphasis': 'reinforce_patterns',
            'success_metric': 'improvement_over_base'
        }

    elif hypothesis == 'awakening':
        # E1 showed no difference
        return {
            'conditions': [
                'base_temporal', 'base_semantic',
                'finetuned_s_semantic', 'finetuned_r_semantic'
            ],
            'focus': 'differentiation',
            'training_emphasis': 'contrast_strategies',
            'success_metric': 'created_advantage'
        }

    elif hypothesis == 'correction':
        # E1 showed reverse (temporal better)
        return {
            'conditions': [
                'base_best', 'finetuned_s_corrected', 'finetuned_r_corrected'
            ],
            'focus': 'bias_correction',
            'training_emphasis': 'fix_misconceptions',
            'success_metric': 'reduced_disadvantage'
        }

    elif hypothesis == 'stabilization':
        # E1 showed high variance
        return {
            'conditions': ['base', 'finetuned_s', 'finetuned_r'],
            'focus': 'consistency',
            'training_emphasis': 'reduce_variance',
            'success_metric': 'variance_reduction'
        }
```

---

## Training Data Generation

### 1. Data Format Specifications

#### 1.1 Summary Format (Dataset S)
```python
def create_summary_dataset(solved_problems):
    """
    Create distilled summaries focusing on retrieval strategy.
    """

    dataset_s = []

    for problem in solved_problems:
        # Analyze solution trajectory
        pattern = identify_pattern(problem)
        key_insight = extract_key_insight(problem)
        retrieval_strategy = determine_optimal_retrieval(problem)

        training_example = {
            'instruction': """Given data points for an unknown equation,
                           analyze the pattern and use memory to solve.""",

            'input': f"Data points: {problem['data_points']}",

            'output': f"""
Step 1 - Initial Analysis:
{pattern['description']}
This suggests {pattern['equation_family']}.

Step 2 - Memory Query Strategy:
Based on {pattern['key_features']}, I should search for:
- Problems with {pattern['similarity_criteria']}
- Examples showing {pattern['growth_pattern']}

Step 3 - Retrieved Examples:
[Simulated retrieval of {pattern['equation_family']} problems]

Step 4 - Solution Approach:
The retrieved examples confirm {key_insight}.
Testing equation form: {pattern['template']}

Step 5 - Final Answer:
{problem['solution']}
            """
        }

        dataset_s.append(training_example)

    return dataset_s
```

#### 1.2 Raw Log Format (Dataset R)
```python
def create_raw_dataset(solved_problems):
    """
    Create training data from complete solution trajectories.
    """

    dataset_r = []

    for problem in solved_problems:
        # Get full solution log
        full_log = problem['solution_trajectory']

        training_example = {
            'instruction': """Given data points for an unknown equation,
                           analyze the pattern and use memory to solve.""",

            'input': f"Data points: {problem['data_points']}",

            'output': format_full_trajectory(full_log)
        }

        dataset_r.append(training_example)

    return dataset_r

def format_full_trajectory(log):
    """Format complete solution trajectory including failures."""

    output = ""
    for attempt in log['attempts']:
        output += f"""
Attempt {attempt['number']}:
Thinking: {attempt['reasoning']}
Query: "{attempt['retrieval_query']}"
Retrieved: {attempt['retrieved_summaries']}
Hypothesis: {attempt['equation_guess']}
Result: Loss = {attempt['loss']:.3f}
"""

        if attempt['success']:
            output += f"\nSuccess! Final equation: {attempt['equation_guess']}"
            break
        else:
            output += f"Analysis: {attempt['failure_analysis']}\n"

    return output
```

#### 1.3 Adaptive Training Data (Based on E1)
```python
def create_adaptive_dataset(solved_problems, e1_hypothesis):
    """
    Create training data adapted to E1 findings.
    """

    if e1_hypothesis == 'awakening':
        # E1 showed no difference - emphasize contrasts
        return create_contrastive_dataset(solved_problems)

    elif e1_hypothesis == 'correction':
        # E1 showed wrong bias - include corrections
        return create_corrective_dataset(solved_problems)

    elif e1_hypothesis == 'stabilization':
        # E1 showed high variance - focus on consistency
        return create_consistent_dataset(solved_problems)

    else:
        # Default: amplification
        return create_reinforcement_dataset(solved_problems)

def create_contrastive_dataset(solved_problems):
    """
    Training data that contrasts retrieval strategies.
    """

    dataset = []

    for problem in solved_problems:
        example = {
            'instruction': "Compare retrieval strategies and choose the best.",

            'input': f"Data points: {problem['data_points']}",

            'output': f"""
Temporal Retrieval Would Give:
{problem['temporal_retrieval_results']}
This is suboptimal because {problem['temporal_weakness']}.

Semantic Retrieval Would Give:
{problem['semantic_retrieval_results']}
This is better because {problem['semantic_strength']}.

Therefore, using semantic retrieval:
Query: "{problem['optimal_query']}"
Solution: {problem['solution']}
            """
        }

        dataset.append(example)

    return dataset
```

### 2. Training Data Statistics

```python
def analyze_training_data(dataset_s, dataset_r):
    """
    Compute statistics on training datasets.
    """

    stats = {
        'dataset_s': {
            'num_examples': len(dataset_s),
            'avg_output_length': np.mean([len(ex['output']) for ex in dataset_s]),
            'total_tokens': sum(count_tokens(ex) for ex in dataset_s),
            'unique_patterns': count_unique_patterns(dataset_s),
            'file_size_mb': get_file_size(dataset_s) / 1024 / 1024
        },
        'dataset_r': {
            'num_examples': len(dataset_r),
            'avg_output_length': np.mean([len(ex['output']) for ex in dataset_r]),
            'total_tokens': sum(count_tokens(ex) for ex in dataset_r),
            'avg_attempts_shown': np.mean([count_attempts(ex) for ex in dataset_r]),
            'file_size_mb': get_file_size(dataset_r) / 1024 / 1024
        }
    }

    stats['size_ratio'] = stats['dataset_r']['file_size_mb'] / \
                         stats['dataset_s']['file_size_mb']
    stats['token_ratio'] = stats['dataset_r']['total_tokens'] / \
                          stats['dataset_s']['total_tokens']

    return stats
```

---

## Fine-tuning Protocol

### 1. LoRA Configuration

```python
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    def __init__(self, conservative=False):
        if conservative:
            # For preservation (from P3 validation)
            self.r = 4
            self.alpha = 8
            self.dropout = 0.1
            self.learning_rate = 1e-5
        else:
            # Standard configuration
            self.r = 8
            self.alpha = 16
            self.dropout = 0.1
            self.learning_rate = 5e-5

        self.target_modules = ["q_proj", "v_proj"]
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.warmup_ratio = 0.1
        self.max_epochs = 3
        self.early_stopping_patience = 3
        self.seed = 42

    def to_dict(self):
        return self.__dict__
```

### 2. Training Implementation

```python
def finetune_model(dataset, config, model_suffix):
    """
    Fine-tune Phi-3-mini using mlx LoRA.
    """

    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, train

    # Setup
    model_name = "phi-3-mini-4k-instruct"
    output_dir = f"models/finetuned_{model_suffix}"

    # Prepare data
    train_data, val_data = split_dataset(dataset, val_ratio=0.2)

    # Training arguments
    training_args = {
        'model': model_name,
        'train': train_data,
        'valid': val_data,
        'iters': len(train_data) // config.batch_size * config.max_epochs,
        'val_iters': len(val_data) // config.batch_size,
        'learning_rate': config.learning_rate,
        'warmup': config.warmup_ratio,
        'batch_size': config.batch_size,
        'lora_layers': config.r,
        'lora_alpha': config.alpha,
        'lora_dropout': config.dropout,
        'adapter_path': output_dir,
        'seed': config.seed
    }

    # Training loop with monitoring
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'early_stop_counter': 0
    }

    def train_with_monitoring():
        start_time = time.time()

        for epoch in range(config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{config.max_epochs}")

            # Training
            train_loss = train_epoch(training_args)
            training_log['train_loss'].append(train_loss)

            # Validation
            val_loss = validate(training_args)
            training_log['val_loss'].append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < training_log['best_val_loss']:
                training_log['best_val_loss'] = val_loss
                training_log['early_stop_counter'] = 0
                save_checkpoint(output_dir, epoch)
            else:
                training_log['early_stop_counter'] += 1

            if training_log['early_stop_counter'] >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        training_log['total_time'] = training_time

        return training_log

    # Execute training
    log = train_with_monitoring()

    # Save final model and logs
    save_model(output_dir)
    save_training_log(log, f"{output_dir}/training_log.json")

    return output_dir, log
```

### 3. Training Monitoring

```python
def monitor_training(model_dir):
    """
    Real-time monitoring of training progress.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update_plots(frame):
        # Load latest log
        with open(f"{model_dir}/training_log.json") as f:
            log = json.load(f)

        # Clear and replot
        ax1.clear()
        ax1.plot(log['train_loss'], label='Train Loss')
        ax1.plot(log['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()

        # Learning rate schedule
        ax2.clear()
        epochs = len(log['train_loss'])
        lrs = [get_lr_at_epoch(e, log['config']) for e in range(epochs)]
        ax2.plot(lrs)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')

    ani = FuncAnimation(fig, update_plots, interval=5000)  # Update every 5s
    plt.show()
```

---

## Evaluation Protocol

### 1. Test Configuration

```python
class E2EvaluationConfig:
    """Configuration for E2 evaluation."""

    def __init__(self, e1_results, e2_design):
        # Use same test set as E1
        self.test_set_path = '../e1-memory-retrieval/data/test_set.json'

        # Use E1's memory database
        self.memory_db_path = '../e1-memory-retrieval/data/memory.db'

        # Retrieval configuration from E1
        self.retrieval_strategy = e1_results['best_strategy']
        self.retrieval_k = 5

        # Conditions to test (varies by hypothesis)
        self.conditions = e2_design['conditions']

        # Evaluation parameters
        self.max_attempts = 20
        self.runs_per_problem = 3
        self.random_seeds = [100, 200, 300]

        # Metrics to collect
        self.metrics = [
            'attempts_to_solve',
            'success_rate',
            'tokens_used',
            'retrieval_query_quality',
            'hypothesis_quality',
            'retrieval_utilization'
        ]
```

### 2. Model Loading

```python
def load_models(e2_config):
    """
    Load all models for evaluation.
    """

    models = {}

    # Base model
    models['base'] = load_base_model('phi-3-mini-4k-instruct')

    # Fine-tuned models
    models['finetuned_s'] = load_with_lora(
        'phi-3-mini-4k-instruct',
        'models/finetuned_summary/adapter.bin'
    )

    models['finetuned_r'] = load_with_lora(
        'phi-3-mini-4k-instruct',
        'models/finetuned_raw/adapter.bin'
    )

    return models
```

### 3. Evaluation Loop

```python
def evaluate_e2(models, test_set, e2_config):
    """
    Main E2 evaluation loop.
    """

    results = []

    for i, problem in enumerate(test_set):
        print(f"\nProblem {i+1}/{len(test_set)}: {problem['problem_id']}")

        problem_results = {
            'problem_id': problem['problem_id'],
            'equation_type': problem['equation_type'],
            'true_equation': problem['true_equation']
        }

        # Test each model condition
        for condition_name in e2_config.conditions:
            model = models[get_model_for_condition(condition_name)]
            retrieval_type = get_retrieval_for_condition(condition_name)

            condition_results = {
                'attempts': [],
                'tokens': [],
                'query_quality': [],
                'hypothesis_quality': []
            }

            # Multiple runs
            for run_idx, seed in enumerate(e2_config.random_seeds):
                result = evaluate_single_run(
                    model, problem, retrieval_type, seed, e2_config
                )

                condition_results['attempts'].append(result['attempts'])
                condition_results['tokens'].append(result['total_tokens'])
                condition_results['query_quality'].append(
                    result['query_quality_score']
                )
                condition_results['hypothesis_quality'].append(
                    result['hypothesis_quality_score']
                )

            # Aggregate results
            problem_results[f'{condition_name}_median_attempts'] = \
                np.median(condition_results['attempts'])
            problem_results[f'{condition_name}_mean_tokens'] = \
                np.mean(condition_results['tokens'])
            problem_results[f'{condition_name}_query_quality'] = \
                np.mean(condition_results['query_quality'])
            problem_results[f'{condition_name}_hypothesis_quality'] = \
                np.mean(condition_results['hypothesis_quality'])

        results.append(problem_results)

        # Checkpoint
        if (i + 1) % 10 == 0:
            save_checkpoint(results, f'e2_checkpoint_{i+1}.json')

    return pd.DataFrame(results)
```

### 4. Quality Metrics

```python
def evaluate_query_quality(query, problem, memory_db):
    """
    Evaluate the quality of a retrieval query.
    """

    # Embed query
    query_embedding = embed_text(query)

    # Get retrieved memories
    retrieved = retrieve_from_db(query_embedding, memory_db, k=5)

    # Compute quality metrics
    metrics = {
        'type_match_rate': sum(1 for m in retrieved
                               if m['equation_type'] == problem['equation_type']) / 5,

        'semantic_diversity': len(set(m['equation_type'] for m in retrieved)) / 5,

        'query_specificity': len(query.split()) / 10,  # Normalized

        'contains_math_terms': any(term in query.lower()
                                   for term in ['linear', 'quadratic',
                                               'exponential', 'periodic'])
    }

    # Weighted combination
    quality_score = (
        0.4 * metrics['type_match_rate'] +
        0.2 * metrics['semantic_diversity'] +
        0.2 * metrics['query_specificity'] +
        0.2 * metrics['contains_math_terms']
    )

    return quality_score, metrics

def evaluate_hypothesis_quality(hypothesis, problem):
    """
    Evaluate the quality of an initial hypothesis.
    """

    true_type = problem['equation_type']

    # Check if hypothesis matches true type
    hypothesis_lower = hypothesis.lower()

    type_keywords = {
        'linear': ['linear', 'straight', 'constant rate'],
        'quadratic': ['quadratic', 'squared', 'parabola', 'x^2', 'x**2'],
        'exponential': ['exponential', 'growth', 'decay', 'compound'],
        'trigonometric': ['sin', 'cos', 'periodic', 'wave', 'oscillat']
    }

    correct_type = any(keyword in hypothesis_lower
                       for keyword in type_keywords[true_type])

    # Check for specific insights
    mentions_pattern = 'pattern' in hypothesis_lower
    mentions_growth = any(term in hypothesis_lower
                         for term in ['increase', 'decrease', 'grow', 'decay'])
    quantitative = any(char.isdigit() for char in hypothesis)

    score = (
        0.5 * correct_type +
        0.2 * mentions_pattern +
        0.2 * mentions_growth +
        0.1 * quantitative
    )

    return score
```

---

## Statistical Analysis

### 1. Hypothesis-Specific Analyses

```python
def analyze_e2_results(results_df, e2_hypothesis):
    """
    Analyze results based on selected hypothesis.
    """

    if e2_hypothesis == 'amplification':
        return analyze_amplification(results_df)
    elif e2_hypothesis == 'awakening':
        return analyze_awakening(results_df)
    elif e2_hypothesis == 'correction':
        return analyze_correction(results_df)
    elif e2_hypothesis == 'stabilization':
        return analyze_stabilization(results_df)

def analyze_awakening(results_df):
    """
    Analysis for awakening hypothesis (creating advantage).
    """

    analysis = {}

    # Test if fine-tuning creates difference
    conditions = ['base_temporal', 'base_semantic',
                  'finetuned_s_semantic', 'finetuned_r_semantic']

    # 1. Baseline comparison (should show no difference)
    stat_base, p_base = wilcoxon(
        results_df['base_temporal_median_attempts'],
        results_df['base_semantic_median_attempts']
    )
    analysis['baseline_difference'] = {
        'p_value': p_base,
        'significant': p_base < 0.05,
        'interpretation': 'No difference expected'
    }

    # 2. Fine-tuned vs Base (should show improvement)
    comparisons = [
        ('finetuned_s_semantic', 'base_semantic', 'Summary Model'),
        ('finetuned_r_semantic', 'base_semantic', 'Raw Model'),
        ('finetuned_s_semantic', 'base_temporal', 'Summary vs Temporal'),
        ('finetuned_r_semantic', 'base_temporal', 'Raw vs Temporal')
    ]

    analysis['finetuned_comparisons'] = {}
    for ft_cond, base_cond, label in comparisons:
        stat, p = wilcoxon(
            results_df[f'{ft_cond}_median_attempts'],
            results_df[f'{base_cond}_median_attempts'],
            alternative='less'
        )

        improvement = (
            results_df[f'{base_cond}_median_attempts'].mean() -
            results_df[f'{ft_cond}_median_attempts'].mean()
        ) / results_df[f'{base_cond}_median_attempts'].mean()

        analysis['finetuned_comparisons'][label] = {
            'p_value': p,
            'significant': p < 0.0125,  # Bonferroni
            'improvement_percent': improvement * 100
        }

    # 3. Success declaration
    analysis['awakening_successful'] = any(
        comp['significant'] and comp['improvement_percent'] > 25
        for comp in analysis['finetuned_comparisons'].values()
    )

    return analysis
```

### 2. Cost-Efficiency Analysis

```python
def analyze_cost_efficiency(results_df, training_logs):
    """
    Comprehensive cost-efficiency analysis.
    """

    # Training costs
    training_costs = {
        'summary': {
            'time_hours': training_logs['summary']['total_time'] / 3600,
            'tokens_millions': training_logs['summary']['total_tokens'] / 1e6,
            'cost_usd': training_logs['summary']['total_tokens'] * 0.003 / 1e6
        },
        'raw': {
            'time_hours': training_logs['raw']['total_time'] / 3600,
            'tokens_millions': training_logs['raw']['total_tokens'] / 1e6,
            'cost_usd': training_logs['raw']['total_tokens'] * 0.003 / 1e6
        }
    }

    # Inference costs (per 1000 problems)
    inference_costs = {}
    for model in ['base', 'finetuned_s', 'finetuned_r']:
        mean_tokens = results_df[f'{model}_mean_tokens'].mean()
        inference_costs[model] = {
            'tokens_per_problem': mean_tokens,
            'cost_per_1k_problems': mean_tokens * 1000 * 0.003 / 1e6,
            'time_per_problem': results_df[f'{model}_time'].mean()
        }

    # Performance metrics
    performance = {}
    for model in ['base', 'finetuned_s', 'finetuned_r']:
        performance[model] = {
            'median_attempts': results_df[f'{model}_median_attempts'].median(),
            'success_rate': (results_df[f'{model}_median_attempts'] < 20).mean(),
            'efficiency': 1 / results_df[f'{model}_median_attempts'].mean()
        }

    # Break-even analysis
    breakeven = {}
    for ft_model in ['finetuned_s', 'finetuned_r']:
        training_cost = training_costs[ft_model.split('_')[1]]['cost_usd']
        inference_savings_per_1k = (
            inference_costs['base']['cost_per_1k_problems'] -
            inference_costs[ft_model]['cost_per_1k_problems']
        )

        if inference_savings_per_1k > 0:
            breakeven[ft_model] = training_cost / inference_savings_per_1k * 1000
        else:
            breakeven[ft_model] = float('inf')

    return {
        'training_costs': training_costs,
        'inference_costs': inference_costs,
        'performance': performance,
        'breakeven_problems': breakeven
    }
```

### 3. Success Criteria Evaluation

```python
def evaluate_success_criteria(cost_efficiency, performance_analysis):
    """
    Check if pre-specified success criteria are met.
    """

    criteria = {
        'efficiency_moderate': False,
        'efficiency_strong': False,
        'performance_moderate': False,
        'performance_strong': False,
        'token_efficiency': False
    }

    # Training time comparison
    time_ratio_s = (cost_efficiency['training_costs']['summary']['time_hours'] /
                    cost_efficiency['training_costs']['raw']['time_hours'])

    # Performance comparison
    perf_ratio_s = (performance_analysis['finetuned_s']['efficiency'] /
                    performance_analysis['finetuned_r']['efficiency'])

    # Token comparison
    token_ratio_s = (cost_efficiency['inference_costs']['finetuned_s']['tokens_per_problem'] /
                     cost_efficiency['inference_costs']['finetuned_r']['tokens_per_problem'])

    # Check criteria
    if time_ratio_s < 0.65 and perf_ratio_s >= 0.95:
        criteria['efficiency_moderate'] = True

    if time_ratio_s < 0.50 and perf_ratio_s >= 0.85:
        criteria['efficiency_strong'] = True

    if perf_ratio_s > 1.25 and time_ratio_s <= 1.5:
        criteria['performance_moderate'] = True

    if perf_ratio_s > 1.50 and time_ratio_s <= 2.0:
        criteria['performance_strong'] = True

    if token_ratio_s < 0.60 and perf_ratio_s >= 0.90:
        criteria['token_efficiency'] = True

    # Overall success
    criteria['any_success'] = any(criteria.values())

    return criteria
```

---

## Visualization Suite

```python
def create_e2_visualizations(results_df, analysis, hypothesis):
    """
    Create comprehensive E2 visualization suite.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Figure layout depends on hypothesis
    if hypothesis == 'awakening':
        fig = create_awakening_figure(results_df, analysis)
    elif hypothesis == 'amplification':
        fig = create_amplification_figure(results_df, analysis)
    else:
        fig = create_standard_figure(results_df, analysis)

    plt.savefig('e2_results.png', dpi=300, bbox_inches='tight')

def create_awakening_figure(results_df, analysis):
    """
    Visualizations for awakening hypothesis.
    """

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 1. Base model comparison (should show no difference)
    ax1 = axes[0, 0]
    data = pd.DataFrame({
        'Temporal': results_df['base_temporal_median_attempts'],
        'Semantic': results_df['base_semantic_median_attempts']
    })
    data.boxplot(ax=ax1)
    ax1.set_title('Base Model: No Difference Expected')
    ax1.set_ylabel('Attempts')

    # 2. Fine-tuned comparison (should show difference)
    ax2 = axes[0, 1]
    data = pd.DataFrame({
        'Base Semantic': results_df['base_semantic_median_attempts'],
        'FT Summary': results_df['finetuned_s_semantic_median_attempts'],
        'FT Raw': results_df['finetuned_r_semantic_median_attempts']
    })
    data.boxplot(ax=ax2)
    ax2.set_title('Fine-tuning Creates Advantage')
    ax2.set_ylabel('Attempts')

    # 3. Query quality improvement
    ax3 = axes[0, 2]
    models = ['Base', 'FT-Summary', 'FT-Raw']
    query_qualities = [
        results_df['base_semantic_query_quality'].mean(),
        results_df['finetuned_s_semantic_query_quality'].mean(),
        results_df['finetuned_r_semantic_query_quality'].mean()
    ]
    ax3.bar(models, query_qualities)
    ax3.set_title('Query Quality Improvement')
    ax3.set_ylabel('Quality Score')

    # 4. Hypothesis quality
    ax4 = axes[1, 0]
    hypothesis_qualities = [
        results_df['base_semantic_hypothesis_quality'].mean(),
        results_df['finetuned_s_semantic_hypothesis_quality'].mean(),
        results_df['finetuned_r_semantic_hypothesis_quality'].mean()
    ]
    ax4.bar(models, hypothesis_qualities)
    ax4.set_title('Hypothesis Quality')
    ax4.set_ylabel('Quality Score')

    # 5. Cost-efficiency frontier
    ax5 = axes[1, 1]
    training_times = [0, analysis['training_costs']['summary']['time_hours'],
                      analysis['training_costs']['raw']['time_hours']]
    performances = [analysis['performance'][m]['efficiency']
                   for m in ['base', 'finetuned_s', 'finetuned_r']]
    ax5.scatter(training_times, performances, s=100)
    for i, model in enumerate(['Base', 'Summary', 'Raw']):
        ax5.annotate(model, (training_times[i], performances[i]))
    ax5.set_xlabel('Training Time (hours)')
    ax5.set_ylabel('Performance (1/attempts)')
    ax5.set_title('Cost-Performance Trade-off')

    # 6. Token usage
    ax6 = axes[1, 2]
    token_usage = [analysis['inference_costs'][m]['tokens_per_problem']
                  for m in ['base', 'finetuned_s', 'finetuned_r']]
    ax6.bar(models, token_usage)
    ax6.set_title('Token Usage per Problem')
    ax6.set_ylabel('Tokens')

    # 7. Break-even analysis
    ax7 = axes[2, 0]
    breakeven_data = pd.DataFrame({
        'Model': ['Summary', 'Raw'],
        'Problems': [analysis['breakeven_problems']['finetuned_s'],
                    analysis['breakeven_problems']['finetuned_r']]
    })
    ax7.bar(breakeven_data['Model'], breakeven_data['Problems'])
    ax7.set_title('Break-even Point')
    ax7.set_ylabel('Problems to Break Even')
    ax7.axhline(y=10000, color='r', linestyle='--', label='10K threshold')
    ax7.legend()

    # 8. Success criteria
    ax8 = axes[2, 1]
    criteria_met = sum(1 for v in analysis['success_criteria'].values() if v)
    criteria_total = len(analysis['success_criteria'])
    ax8.pie([criteria_met, criteria_total - criteria_met],
           labels=['Met', 'Not Met'],
           autopct='%1.1f%%')
    ax8.set_title('Success Criteria')

    # 9. Final comparison
    ax9 = axes[2, 2]
    final_comparison = pd.DataFrame({
        'Metric': ['Attempts', 'Tokens', 'Training Time'],
        'Summary vs Raw': [
            analysis['performance']['finetuned_s']['median_attempts'] /
            analysis['performance']['finetuned_r']['median_attempts'],
            analysis['inference_costs']['finetuned_s']['tokens_per_problem'] /
            analysis['inference_costs']['finetuned_r']['tokens_per_problem'],
            analysis['training_costs']['summary']['time_hours'] /
            analysis['training_costs']['raw']['time_hours']
        ]
    })
    ax9.barh(final_comparison['Metric'], final_comparison['Summary vs Raw'])
    ax9.axvline(x=1, color='r', linestyle='--')
    ax9.set_xlabel('Summary / Raw Ratio')
    ax9.set_title('Summary vs Raw Comparison')

    plt.suptitle(f'E2 Results: {hypothesis.capitalize()} Hypothesis', fontsize=16)
    plt.tight_layout()

    return fig
```

---

## Output Specifications

### Files Generated
```
e2-fine-tuning/
├── data/
│   ├── dataset_s.json          # Summary training data
│   ├── dataset_r.json          # Raw log training data
│   └── training_stats.json     # Data statistics
├── models/
│   ├── finetuned_summary/
│   │   ├── adapter.bin         # LoRA weights
│   │   └── training_log.json   # Training history
│   └── finetuned_raw/
│       ├── adapter.bin
│       └── training_log.json
├── results/
│   ├── raw_results.csv         # All evaluation data
│   ├── analysis_summary.json   # Statistical analysis
│   ├── cost_efficiency.json    # Cost analysis
│   ├── success_criteria.json   # Criteria evaluation
│   └── hypothesis_result.json  # Main finding
├── visualizations/
│   ├── e2_results.png          # Main results figure
│   ├── training_curves.png     # Training progress
│   ├── cost_frontier.png       # Efficiency analysis
│   └── query_quality.png       # Query improvement
└── logs/
    └── experiment.log           # Detailed execution log
```

### Key Deliverables

1. **Main Finding**
   ```json
   {
     "hypothesis_tested": "awakening",
     "hypothesis_confirmed": true,
     "key_result": "Fine-tuning creates 35% advantage where none existed",
     "best_model": "finetuned_summary",
     "improvement_over_base": "35%",
     "training_roi": "Breaks even at 8,500 problems"
   }
   ```

2. **Recommendation**
   ```json
   {
     "deploy_model": "finetuned_summary",
     "rationale": "Better cost-efficiency ratio",
     "expected_savings": "$0.15 per 1000 problems",
     "confidence": "high (p < 0.001)"
   }
   ```

---

## Timeline

### Day 1: Data Preparation (6 hours)
- Load E1 results and determine hypothesis
- Generate Dataset S (summaries)
- Generate Dataset R (raw logs)
- Analyze data statistics

### Day 2: Fine-tuning (8 hours)
- Fine-tune Model-S (3 hours)
- Fine-tune Model-R (4 hours)
- Validate models (1 hour)

### Day 3: Evaluation (8 hours)
- Run evaluation on all models
- 150 problems × conditions × 3 runs
- Collect all metrics

### Day 4: Analysis (6 hours)
- Statistical analysis
- Cost-efficiency analysis
- Generate visualizations
- Write final report

---

## Success Indicators

1. **Scientific Success**
   - Clear result for tested hypothesis (p < 0.05)
   - Effect size > 0.3 (meaningful improvement)
   - Consistent results across problem types

2. **Practical Success**
   - At least one success criterion met
   - Break-even < 50,000 problems
   - Deployment recommendation clear

3. **Technical Success**
   - Models converge during training
   - <5% evaluation failures
   - Reproducible results