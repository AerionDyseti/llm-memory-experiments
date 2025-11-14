# E1: Memory Retrieval Strategies Experiment

## Overview
This experiment compares three retrieval strategies (none, temporal, semantic) to establish baseline performance and determine which E2 hypothesis to test. This is both a standalone experiment and a critical prerequisite for E2.

## Research Questions
1. Does retrieval strategy significantly impact problem-solving efficiency?
2. Which retrieval method provides the best baseline performance?
3. Is there room for improvement that fine-tuning could exploit?

## Hypotheses
- **H₁**: Semantic retrieval will outperform temporal retrieval (attempts_semantic < attempts_temporal)
- **H₂**: Both retrieval methods will outperform no retrieval (attempts_retrieval < attempts_none)
- **H₃**: Retrieval quality will correlate with problem-solving success (r > 0.3)

---

## Experimental Design

### Within-Subjects Design
- Each of 150 test problems is evaluated under all 3 conditions
- Controls for problem difficulty variation
- Enables paired statistical tests
- Order randomized to prevent learning effects

### Conditions
1. **No Memory** (Baseline): Agent solves with only current problem data
2. **Temporal Retrieval** (Control): Retrieves 5 most recent memories
3. **Semantic Retrieval** (Experimental): Retrieves 5 most similar memories

---

## Methodology

### 1. Data Generation

#### 1.1 Problem Space Definition
```python
class ProblemGenerator:
    """
    Generates mathematical problems with known solutions.
    Ensures no overlap between memory and test sets.
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.used_equations = set()  # Track to prevent duplicates

    def generate_problem(self, equation_type, problem_id):
        """Generate a single problem with guaranteed uniqueness."""

        equation = None
        attempts = 0

        while equation is None or equation in self.used_equations:
            if equation_type == 'linear':
                a = self.rng.uniform(-10, 10)
                b = self.rng.uniform(-20, 20)
                equation = f"{a:.2f}*x + {b:.2f}"

            elif equation_type == 'quadratic':
                a = self.rng.uniform(-5, 5)
                b = self.rng.uniform(-5, 5)
                c = self.rng.uniform(-5, 5)
                equation = f"{a:.2f}*x**2 + {b:.2f}*x + {c:.2f}"

            elif equation_type == 'exponential':
                a = self.rng.uniform(1, 5)
                b = self.rng.uniform(1.1, 3)
                equation = f"{a:.2f} * {b:.2f}**x"

            elif equation_type == 'trigonometric':
                a = self.rng.uniform(1, 10)
                b = self.rng.uniform(0.5, 2)
                c = self.rng.uniform(-5, 5)
                equation = f"{a:.2f} * sin({b:.2f}*x) + {c:.2f}"

            attempts += 1
            if attempts > 100:
                raise ValueError(f"Cannot generate unique {equation_type}")

        self.used_equations.add(equation)

        # Generate noisy data points
        x_values = self.rng.uniform(-10, 10, 10)
        x_values.sort()

        # Safely evaluate equation
        y_values = []
        for x in x_values:
            try:
                # Use numexpr for safe evaluation
                y = eval(equation.replace('sin', 'np.sin'),
                         {'x': x, 'np': np, '__builtins__': {}})
                y_values.append(y)
            except:
                y_values.append(0)  # Fallback

        # Add noise
        noise = self.rng.normal(0, 0.1 * np.std(y_values), len(y_values))
        y_noisy = np.array(y_values) + noise

        return {
            'problem_id': problem_id,
            'equation_type': equation_type,
            'true_equation': equation,
            'data_points': list(zip(x_values.tolist(), y_noisy.tolist())),
            'metadata': {
                'x_range': [float(x_values.min()), float(x_values.max())],
                'y_range': [float(y_noisy.min()), float(y_noisy.max())],
                'noise_level': float(np.std(noise))
            }
        }

    def generate_dataset(self, n_problems, distribution=None):
        """Generate full dataset with specified type distribution."""

        if distribution is None:
            distribution = {
                'linear': 0.25,
                'quadratic': 0.25,
                'exponential': 0.25,
                'trigonometric': 0.25
            }

        problems = []
        for equation_type, proportion in distribution.items():
            n_type = int(n_problems * proportion)
            for i in range(n_type):
                problem_id = f"{equation_type}_{i:04d}"
                problem = self.generate_problem(equation_type, problem_id)
                problems.append(problem)

        # Shuffle to avoid type clustering
        self.rng.shuffle(problems)
        return problems
```

#### 1.2 Dataset Creation
```python
def create_datasets():
    """Create memory and test sets with validation."""

    generator = ProblemGenerator(seed=42)

    # Memory set: 1500 problems
    memory_set = generator.generate_dataset(1500)

    # Test set: 150 problems (non-overlapping)
    test_set = generator.generate_dataset(150)

    # Validation: Ensure no equation overlap
    memory_equations = {p['true_equation'] for p in memory_set}
    test_equations = {p['true_equation'] for p in test_set}

    overlap = memory_equations & test_equations
    if overlap:
        raise ValueError(f"Equation overlap detected: {overlap}")

    # Save datasets
    with open('memory_set.json', 'w') as f:
        json.dump(memory_set, f, indent=2)

    with open('test_set.json', 'w') as f:
        json.dump(test_set, f, indent=2)

    return memory_set, test_set
```

### 2. Memory Database Creation

#### 2.1 Baseline Agent for Memory Population
```python
class BaselineAgent:
    """
    Simple agent to solve problems and create memory entries.
    """

    def __init__(self, model='phi-3-mini-instruct', temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.ollama = ollama.Client()

    def solve_problem(self, problem, max_attempts=20):
        """Attempt to solve a problem, return solution memo."""

        attempts = []
        success = False

        for attempt_num in range(max_attempts):
            # Generate guess
            prompt = self.create_prompt(problem, attempts)
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature
            )

            guess = self.extract_equation(response['response'])
            loss = self.calculate_loss(guess, problem['data_points'])

            attempts.append({
                'attempt': attempt_num + 1,
                'guess': guess,
                'loss': loss
            })

            if loss < 0.1:  # Success threshold
                success = True
                break

        # Create memory memo
        memo = {
            'problem_id': problem['problem_id'],
            'equation_type': problem['equation_type'],
            'data_summary': self.summarize_data(problem['data_points']),
            'solution': guess if success else problem['true_equation'],
            'attempts_taken': len(attempts),
            'success': success,
            'failed_guesses': [a['guess'] for a in attempts[:-1]],
            'strategy': self.infer_strategy(attempts)
        }

        return memo

    def create_prompt(self, problem, previous_attempts):
        """Create prompt for equation fitting."""

        prompt = f"""
        Find the equation that fits these data points:
        {problem['data_points']}

        Previous attempts: {len(previous_attempts)}
        """

        if previous_attempts:
            prompt += "\nFailed guesses:\n"
            for attempt in previous_attempts[-3:]:  # Show last 3
                prompt += f"  {attempt['guess']} (loss: {attempt['loss']:.3f})\n"

        prompt += "\nProvide your next guess as a Python expression using x."

        return prompt
```

#### 2.2 Database Setup
```python
def create_memory_database(memory_set):
    """
    Create SQLite database with vector extension for embeddings.
    """

    # Initialize database
    conn = sqlite3.connect('memory.db')
    conn.enable_load_extension(True)
    conn.load_extension('vec0')  # sqlite-vec

    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            problem_id TEXT UNIQUE,
            equation_type TEXT,
            solution TEXT,
            attempts INTEGER,
            strategy TEXT,
            memo_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
            memory_id INTEGER,
            text_embedding FLOAT[768],  -- MiniLM dimension
            feature_embedding FLOAT[10]  -- Numerical features
        )
    """)

    # Populate database
    agent = BaselineAgent()
    embedding_model = load_embedding_model()

    for i, problem in enumerate(memory_set):
        print(f"Processing problem {i+1}/{len(memory_set)}")

        # Solve problem to create memo
        memo = agent.solve_problem(problem)

        # Insert memo
        cursor = conn.execute("""
            INSERT INTO memories
            (problem_id, equation_type, solution, attempts, strategy, memo_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            memo['problem_id'],
            memo['equation_type'],
            memo['solution'],
            memo['attempts_taken'],
            memo['strategy'],
            json.dumps(memo)
        ))

        memory_id = cursor.lastrowid

        # Create embeddings
        text_emb = embedding_model.encode(memo['data_summary'])
        feature_emb = extract_numerical_features(problem['data_points'])

        # Insert embeddings
        conn.execute("""
            INSERT INTO memory_embeddings
            (memory_id, text_embedding, feature_embedding)
            VALUES (?, ?, ?)
        """, (memory_id, text_emb.tolist(), feature_emb.tolist()))

        # Commit periodically
        if (i + 1) % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()
```

### 3. Experimental Conditions Implementation

#### 3.1 No Memory Condition
```python
class NoMemoryAgent:
    """Agent without access to memory database."""

    def __init__(self, model='phi-3-mini-instruct', temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.ollama = ollama.Client()

    def solve(self, problem, max_attempts=20, run_seed=None):
        """Solve problem without any retrieval."""

        if run_seed:
            np.random.seed(run_seed)

        prompt = f"""
        You are solving an equation fitting problem.

        Data points: {problem['data_points']}

        Analyze the pattern and find the equation.
        Express your answer as a Python expression using x.

        Think step by step:
        1. Look at how y changes with x
        2. Identify the pattern (linear, quadratic, exponential, etc.)
        3. Propose an equation
        """

        attempts = 0
        for i in range(max_attempts):
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt + f"\n\nAttempt {i+1}:",
                temperature=self.temperature
            )

            equation = extract_equation(response['response'])
            loss = calculate_loss(equation, problem['data_points'])

            attempts += 1

            if loss < 0.1:
                break

            prompt += f"\nGuess: {equation}, Loss: {loss:.3f}"

        return {
            'attempts': attempts,
            'final_equation': equation,
            'success': loss < 0.1,
            'condition': 'no_memory'
        }
```

#### 3.2 Temporal Retrieval Condition
```python
class TemporalRetrievalAgent:
    """Agent with temporal (recent) retrieval."""

    def __init__(self, model='phi-3-mini-instruct', temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.ollama = ollama.Client()
        self.conn = sqlite3.connect('memory.db')

    def retrieve_recent(self, k=5):
        """Retrieve k most recent memories."""

        cursor = self.conn.execute("""
            SELECT memo_text
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        """, (k,))

        memories = [json.loads(row[0]) for row in cursor.fetchall()]
        return memories

    def solve(self, problem, max_attempts=20, run_seed=None):
        """Solve with temporal retrieval."""

        if run_seed:
            np.random.seed(run_seed)

        # Retrieve recent memories
        memories = self.retrieve_recent(k=5)
        memory_text = self.format_memories(memories)

        prompt = f"""
        You are solving an equation fitting problem.

        Data points: {problem['data_points']}

        Here are some recently solved problems:
        {memory_text}

        Using insights from these examples, find the equation.
        Express your answer as a Python expression using x.
        """

        attempts = 0
        for i in range(max_attempts):
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt + f"\n\nAttempt {i+1}:",
                temperature=self.temperature
            )

            equation = extract_equation(response['response'])
            loss = calculate_loss(equation, problem['data_points'])

            attempts += 1

            if loss < 0.1:
                break

            prompt += f"\nGuess: {equation}, Loss: {loss:.3f}"

        return {
            'attempts': attempts,
            'final_equation': equation,
            'success': loss < 0.1,
            'condition': 'temporal',
            'retrieved_types': [m['equation_type'] for m in memories]
        }
```

#### 3.3 Semantic Retrieval Condition
```python
class SemanticRetrievalAgent:
    """Agent with semantic (similarity-based) retrieval."""

    def __init__(self, model='phi-3-mini-instruct', temperature=0.3):
        self.model = model
        self.temperature = temperature
        self.ollama = ollama.Client()
        self.conn = sqlite3.connect('memory.db')
        self.conn.enable_load_extension(True)
        self.conn.load_extension('vec0')

        # Load config from E0
        with open('../e0-parameter-optimization/results/e1_config.json') as f:
            self.retrieval_config = json.load(f)

    def retrieve_similar(self, problem, k=5):
        """Retrieve k most similar memories."""

        # Create query embedding
        embedding_model = load_embedding_model(
            self.retrieval_config['embedding_model']
        )

        # Text embedding
        text = f"Data points: {problem['data_points']}"
        text_emb = embedding_model.encode(text)

        # Numerical features
        feature_emb = extract_numerical_features(problem['data_points'])

        # Weighted combination (from E0 optimization)
        text_weight = self.retrieval_config['text_weight']
        feature_weight = self.retrieval_config['feature_weight']

        # Query database using vec0
        cursor = self.conn.execute("""
            SELECT m.memo_text,
                   vec_distance_cosine(
                       me.text_embedding * ? + me.feature_embedding * ?,
                       ? || ?
                   ) as distance
            FROM memories m
            JOIN memory_embeddings me ON m.id = me.memory_id
            ORDER BY distance
            LIMIT ?
        """, (
            text_weight, feature_weight,
            text_emb.tolist(), feature_emb.tolist(),
            k
        ))

        memories = [json.loads(row[0]) for row in cursor.fetchall()]
        return memories

    def solve(self, problem, max_attempts=20, run_seed=None):
        """Solve with semantic retrieval."""

        if run_seed:
            np.random.seed(run_seed)

        # Retrieve similar memories
        memories = self.retrieve_similar(problem, k=5)
        memory_text = self.format_memories(memories)

        prompt = f"""
        You are solving an equation fitting problem.

        Data points: {problem['data_points']}

        Here are similar problems and their solutions:
        {memory_text}

        Using patterns from these similar examples, find the equation.
        Express your answer as a Python expression using x.
        """

        attempts = 0
        for i in range(max_attempts):
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt + f"\n\nAttempt {i+1}:",
                temperature=self.temperature
            )

            equation = extract_equation(response['response'])
            loss = calculate_loss(equation, problem['data_points'])

            attempts += 1

            if loss < 0.1:
                break

            prompt += f"\nGuess: {equation}, Loss: {loss:.3f}"

        return {
            'attempts': attempts,
            'final_equation': equation,
            'success': loss < 0.1,
            'condition': 'semantic',
            'retrieved_types': [m['equation_type'] for m in memories],
            'retrieval_relevance': self.calculate_relevance(
                memories, problem['equation_type']
            )
        }
```

### 4. Evaluation Protocol

#### 4.1 Main Experiment Loop
```python
def run_experiment(test_set, conditions_order='randomized'):
    """
    Run main E1 experiment with all conditions.
    """

    results = []
    agents = {
        'none': NoMemoryAgent(),
        'temporal': TemporalRetrievalAgent(),
        'semantic': SemanticRetrievalAgent()
    }

    # Run each problem through all conditions
    for i, problem in enumerate(test_set):
        print(f"\nProblem {i+1}/{len(test_set)}: {problem['problem_id']}")

        # Randomize condition order per problem
        if conditions_order == 'randomized':
            conditions = ['none', 'temporal', 'semantic']
            np.random.shuffle(conditions)
        else:
            conditions = conditions_order

        problem_results = {
            'problem_id': problem['problem_id'],
            'equation_type': problem['equation_type'],
            'true_equation': problem['true_equation']
        }

        # Test each condition with multiple runs
        for condition in conditions:
            agent = agents[condition]
            condition_attempts = []

            # Run 3 times with different seeds
            for run in range(3):
                run_seed = 100 * (run + 1)
                result = agent.solve(problem, run_seed=run_seed)
                condition_attempts.append(result['attempts'])

            # Store median attempts
            problem_results[f'{condition}_attempts'] = condition_attempts
            problem_results[f'{condition}_median'] = np.median(condition_attempts)
            problem_results[f'{condition}_success'] = result['success']

            # Store retrieval quality metrics if applicable
            if condition in ['temporal', 'semantic']:
                problem_results[f'{condition}_retrieved_types'] = \
                    result.get('retrieved_types', [])

                if condition == 'semantic':
                    problem_results['retrieval_relevance'] = \
                        result.get('retrieval_relevance', 0)

        results.append(problem_results)

        # Save checkpoint every 10 problems
        if (i + 1) % 10 == 0:
            save_checkpoint(results, f'checkpoint_{i+1}.json')

    return pd.DataFrame(results)
```

#### 4.2 Token Usage Tracking
```python
def track_token_usage(agent, problem):
    """
    Track token usage for cost analysis.
    """

    # Intercept ollama calls to count tokens
    original_generate = agent.ollama.generate
    token_counts = {'input': 0, 'output': 0}

    def wrapped_generate(*args, **kwargs):
        prompt = kwargs.get('prompt', args[1] if len(args) > 1 else '')
        response = original_generate(*args, **kwargs)

        # Approximate token count (actual would use tokenizer)
        token_counts['input'] += len(prompt.split()) * 1.3
        token_counts['output'] += len(response['response'].split()) * 1.3

        return response

    agent.ollama.generate = wrapped_generate

    # Run solve
    result = agent.solve(problem)

    # Restore original
    agent.ollama.generate = original_generate

    result['tokens'] = token_counts
    return result
```

### 5. Statistical Analysis

#### 5.1 Primary Analyses
```python
def analyze_results(results_df):
    """
    Comprehensive statistical analysis of E1 results.
    """

    analysis = {}

    # 1. Descriptive Statistics
    conditions = ['none', 'temporal', 'semantic']
    for condition in conditions:
        analysis[f'{condition}_stats'] = {
            'mean': results_df[f'{condition}_median'].mean(),
            'std': results_df[f'{condition}_median'].std(),
            'median': results_df[f'{condition}_median'].median(),
            'success_rate': results_df[f'{condition}_success'].mean(),
            'censored_rate': (results_df[f'{condition}_median'] == 20).mean()
        }

    # 2. Friedman Test (non-parametric for 3+ related samples)
    from scipy.stats import friedmanchisquare
    data_for_friedman = [
        results_df['none_median'].values,
        results_df['temporal_median'].values,
        results_df['semantic_median'].values
    ]
    friedman_stat, friedman_p = friedmanchisquare(*data_for_friedman)
    analysis['friedman'] = {'statistic': friedman_stat, 'p_value': friedman_p}

    # 3. Post-hoc Pairwise Comparisons (Wilcoxon with Bonferroni)
    from scipy.stats import wilcoxon
    alpha_bonferroni = 0.05 / 3  # Three comparisons

    comparisons = [
        ('none', 'temporal'),
        ('none', 'semantic'),
        ('temporal', 'semantic')
    ]

    analysis['pairwise'] = {}
    for cond1, cond2 in comparisons:
        stat, p = wilcoxon(
            results_df[f'{cond1}_median'],
            results_df[f'{cond2}_median'],
            alternative='two-sided'
        )

        # Effect size (Cliff's delta for non-parametric)
        delta = cliff_delta(
            results_df[f'{cond1}_median'],
            results_df[f'{cond2}_median']
        )

        analysis['pairwise'][f'{cond1}_vs_{cond2}'] = {
            'statistic': stat,
            'p_value': p,
            'significant': p < alpha_bonferroni,
            'effect_size': delta,
            'interpretation': interpret_cliff_delta(delta)
        }

    # 4. Retrieval Quality Analysis
    if 'retrieval_relevance' in results_df.columns:
        correlation = results_df[['retrieval_relevance', 'semantic_median']].corr()
        analysis['retrieval_correlation'] = correlation.iloc[0, 1]

    return analysis
```

#### 5.2 E2 Hypothesis Determination
```python
def determine_e2_hypothesis(analysis):
    """
    Based on E1 results, determine which E2 hypothesis to test.
    """

    semantic_vs_temporal = analysis['pairwise']['semantic_vs_temporal']
    semantic_vs_none = analysis['pairwise']['semantic_vs_none']

    # Decision tree
    if semantic_vs_temporal['p_value'] < 0.01:
        if semantic_vs_temporal['effect_size'] > 0:
            hypothesis = 'amplification'
            description = "Semantic >> Temporal: Test if fine-tuning amplifies advantage"
        else:
            hypothesis = 'correction'
            description = "Semantic << Temporal: Test if fine-tuning corrects bias"

    elif semantic_vs_temporal['p_value'] > 0.10:
        hypothesis = 'awakening'
        description = "Semantic ≈ Temporal: Test if fine-tuning creates advantage"

    else:  # 0.01 < p < 0.10
        hypothesis = 'weak_signal'
        description = "Weak difference: Test if fine-tuning strengthens signal"

    # Check for high variance
    cv_semantic = analysis['semantic_stats']['std'] / analysis['semantic_stats']['mean']
    cv_temporal = analysis['temporal_stats']['std'] / analysis['temporal_stats']['mean']

    if max(cv_semantic, cv_temporal) > 0.5:
        hypothesis = 'stabilization'
        description = "High variance: Test if fine-tuning stabilizes performance"

    return {
        'hypothesis': hypothesis,
        'description': description,
        'semantic_vs_temporal_p': semantic_vs_temporal['p_value'],
        'effect_size': semantic_vs_temporal['effect_size'],
        'recommendation': get_e2_design_recommendation(hypothesis)
    }
```

### 6. Visualization Suite

```python
def create_visualizations(results_df, analysis):
    """
    Create comprehensive visualization suite for E1 results.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Violin plot of attempts by condition
    ax1 = fig.add_subplot(gs[0, :])
    data_long = pd.melt(
        results_df[['none_median', 'temporal_median', 'semantic_median']],
        var_name='Condition',
        value_name='Attempts'
    )
    data_long['Condition'] = data_long['Condition'].str.replace('_median', '')
    sns.violinplot(data=data_long, x='Condition', y='Attempts', ax=ax1)
    ax1.set_title('Distribution of Attempts by Retrieval Strategy')
    ax1.set_ylabel('Attempts to Solve')

    # 2. Success rates
    ax2 = fig.add_subplot(gs[1, 0])
    success_rates = [
        analysis['none_stats']['success_rate'],
        analysis['temporal_stats']['success_rate'],
        analysis['semantic_stats']['success_rate']
    ]
    ax2.bar(['None', 'Temporal', 'Semantic'], success_rates)
    ax2.set_title('Success Rates')
    ax2.set_ylabel('Proportion Successful')
    ax2.set_ylim([0, 1])

    # 3. Paired differences
    ax3 = fig.add_subplot(gs[1, 1])
    differences = results_df['semantic_median'] - results_df['temporal_median']
    ax3.hist(differences, bins=20, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', label='No difference')
    ax3.set_title('Semantic - Temporal Differences')
    ax3.set_xlabel('Difference in Attempts')
    ax3.legend()

    # 4. Problem type analysis
    ax4 = fig.add_subplot(gs[1, 2])
    by_type = results_df.groupby('equation_type').agg({
        'none_median': 'mean',
        'temporal_median': 'mean',
        'semantic_median': 'mean'
    })
    by_type.plot(kind='bar', ax=ax4)
    ax4.set_title('Performance by Problem Type')
    ax4.set_ylabel('Mean Attempts')
    ax4.legend(['None', 'Temporal', 'Semantic'])

    # 5. Retrieval relevance vs performance
    if 'retrieval_relevance' in results_df.columns:
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(results_df['retrieval_relevance'],
                   results_df['semantic_median'])
        ax5.set_xlabel('Retrieval Relevance Score')
        ax5.set_ylabel('Attempts (Semantic)')
        ax5.set_title('Retrieval Quality vs Performance')

        # Add regression line
        z = np.polyfit(results_df['retrieval_relevance'],
                      results_df['semantic_median'], 1)
        p = np.poly1d(z)
        ax5.plot(results_df['retrieval_relevance'],
                p(results_df['retrieval_relevance']),
                "r--", alpha=0.5)

    # 6. Token usage comparison
    ax6 = fig.add_subplot(gs[2, 1])
    # Would need token data from tracking
    ax6.text(0.5, 0.5, 'Token Usage\n(Placeholder)',
            ha='center', va='center', fontsize=12)
    ax6.set_title('Token Efficiency')

    # 7. Effect sizes
    ax7 = fig.add_subplot(gs[2, 2])
    comparisons = ['None vs\nTemporal', 'None vs\nSemantic', 'Temporal vs\nSemantic']
    effect_sizes = [
        analysis['pairwise']['none_vs_temporal']['effect_size'],
        analysis['pairwise']['none_vs_semantic']['effect_size'],
        analysis['pairwise']['temporal_vs_semantic']['effect_size']
    ]
    colors = ['green' if abs(es) > 0.3 else 'orange' if abs(es) > 0.1 else 'red'
              for es in effect_sizes]
    ax7.bar(comparisons, effect_sizes, color=colors)
    ax7.set_title('Effect Sizes (Cliff\'s Delta)')
    ax7.set_ylabel('Effect Size')
    ax7.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax7.set_ylim([-1, 1])

    plt.suptitle('E1: Memory Retrieval Strategies Results', fontsize=16)
    plt.savefig('e1_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Output Specifications

### Files Generated
```
e1-memory-retrieval/
├── data/
│   ├── memory_set.json         # 1500 problems for memory
│   ├── test_set.json           # 150 test problems
│   └── memory.db               # SQLite with embeddings
├── results/
│   ├── raw_results.csv         # All attempts data
│   ├── analysis_summary.json   # Statistical analysis
│   ├── e2_hypothesis.json      # E2 recommendation
│   └── token_usage.csv         # Cost analysis
├── visualizations/
│   ├── e1_results.png          # Main results figure
│   ├── by_problem_type.png     # Breakdown by type
│   └── retrieval_quality.png   # Quality analysis
├── checkpoints/
│   └── checkpoint_*.json       # Progress saves
└── logs/
    └── experiment.log           # Detailed execution log
```

### Key Deliverables for E2

1. **Performance Baselines**
   ```json
   {
     "semantic_median_attempts": 7.2,
     "temporal_median_attempts": 9.1,
     "none_median_attempts": 12.3,
     "semantic_success_rate": 0.85,
     "best_strategy": "semantic"
   }
   ```

2. **E2 Hypothesis Selection**
   ```json
   {
     "hypothesis": "awakening",
     "rationale": "No significant difference between strategies",
     "p_value": 0.15,
     "recommended_design": "4-condition with focus on differentiation"
   }
   ```

3. **Token Usage Baseline**
   ```json
   {
     "semantic_tokens_per_problem": 1250,
     "temporal_tokens_per_problem": 1180,
     "none_tokens_per_problem": 980,
     "cost_per_100_problems": "$0.38"
   }
   ```

---

## Execution Timeline

### Day 1: Setup and Data Generation (8 hours)
- Generate 1500 memory problems
- Generate 150 test problems
- Validate non-overlap
- Run baseline agent on memory set
- Create SQLite database with embeddings

### Day 2: Experiment Execution (8 hours)
- Run all three conditions
- 150 problems × 3 conditions × 3 runs = 1350 evaluations
- ~20 seconds per evaluation
- Regular checkpointing

### Day 3: Analysis and Reporting (4 hours)
- Statistical analysis
- E2 hypothesis determination
- Visualization creation
- Report generation

---

## Risk Mitigation

| Risk | Mitigation | Impact |
|------|------------|---------|
| Agent gets stuck | Implement timeout (60s) | Minor delay |
| Database corruption | Regular backups | Could lose progress |
| Memory leaks | Restart every 50 problems | Adds 10 min |
| Inconsistent results | Fix all random seeds | Ensures reproducibility |

---

## Success Criteria

1. **Technical Success**
   - All 150 problems evaluated under all conditions
   - <5% data loss or corruption
   - Reproducible results with fixed seeds

2. **Scientific Success**
   - Clear statistical differences observed (p < 0.05 for at least one comparison)
   - Effect sizes > 0.1 (non-negligible)
   - Retrieval relevance correlates with performance (r > 0.2)

3. **E2 Readiness**
   - Clear hypothesis selected for E2
   - Stable baselines established
   - Token usage characterized