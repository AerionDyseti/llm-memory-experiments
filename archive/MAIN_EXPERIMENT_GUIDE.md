# Main Experiment Guide: LLM Memory Systems

## Quick Start
This guide provides the complete roadmap for executing the LLM memory experiments. All historical design iterations have been archived. Follow this guide sequentially.

**Total Timeline: 7 days** | **Hardware: M2 Pro** | **Key Question: Can fine-tuning create better RAG intuition?**

---

## Experiment Overview

### The Chain
```
Prerequisites (10hrs) → E0 Optimization (8hrs) → E1 Retrieval (2 days) → E2 Fine-tuning (3 days)
```

### The Core Hypothesis
**E2 (Main)**: Fine-tuning can teach models to generate better retrieval queries ("hunches") even when the base model shows no preference between retrieval strategies.

### Key Innovation
E1 results don't block E2 - they determine *what* E2 tests:
- Semantic >> Temporal → Test amplification
- Semantic ≈ Temporal → Test awakening (most interesting!)
- Semantic < Temporal → Test correction
- High variance → Test stabilization

---

## Phase 0: Setup Checklist

### Required Packages
```bash
pip install ollama sqlite-vec sentence-transformers scipy numpy pandas matplotlib
# For M2 Pro neural engine:
pip install mlx mlx-lm
```

### Directory Structure
```
experiment/
├── MAIN_EXPERIMENT_GUIDE.md (this file)
├── experiment_flowchart.md
├── prerequisites/
│   └── methodology.md
├── e0-parameter-optimization/
│   └── methodology.md
├── e1-memory-retrieval/
│   └── methodology.md
├── e2-fine-tuning/
│   └── methodology.md
└── archive/ (historical iterations)
```

### Models Required
- **Primary**: Phi-3-mini-4k-instruct (via Ollama)
- **Embeddings**: all-MiniLM-L6-v2 (via sentence-transformers)

---

## Phase 1: Prerequisites (Day 1, 10 hours)

### Critical Gates - MUST PASS or STOP

#### P1: Embedding Validity (3 hours)
**File**: `prerequisites/methodology.md` → Section P1

**What**: Verify embeddings can distinguish equation types
**Pass Criteria**: Similarity ratio > 1.2
**Fail Action**: STOP - Redesign similarity metric

Quick Test:
```python
# Generate 100 problems (25 each: linear, quadratic, exponential, trig)
# Create embeddings
# Check: intra_class_similarity > inter_class_similarity + 0.3
```

#### P2: Retrieval Utility (4 hours)
**File**: `prerequisites/methodology.md` → Section P2

**What**: Verify that ANY retrieval helps
**Pass Criteria**: p < 0.05 vs no retrieval
**Fail Action**: STOP - Investigate why examples don't help

Quick Test:
```python
# Compare: No retrieval vs Random 5 vs Recent 5
# Run on 30 test problems
# Check: at least one retrieval method helps significantly
```

#### P3: Fine-tuning Preservation (3 hours)
**File**: `prerequisites/methodology.md` → Section P3

**What**: Verify LoRA doesn't break the model
**Pass Criteria**: Retains >80% capabilities
**Fail Action**: STOP - Adjust LoRA parameters

Quick Test:
```python
# Fine-tune on 50 problems aggressively
# Test: instruction following, reasoning, format compliance
# Check: performance drop <20% on all benchmarks
```

### Decision Point
✅ All pass → Continue to E0
❌ Any fail → Stop and address issue

---

## Phase 2: E0 Parameter Optimization (Day 2, 8 hours)

**File**: `e0-parameter-optimization/methodology.md`

### Goal
Find optimal retrieval parameters before main experiments

### Grid Search Space
- **Embedding models**: MiniLM, mpnet, angle-bert
- **Weight ratios**: (text:features) from (1:0) to (0:1)
- **Retrieval counts**: 3, 5, 7, 10
- **Total**: 120 configurations

### Process
1. Create 200-problem validation set
2. Test each configuration (~3 min each)
3. Select based on relevance score
4. Output: `e1_config.json` with optimal parameters

### Key Output for E1
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "text_weight": 0.7,
  "feature_weight": 0.3,
  "retrieval_count": 5,
  "expected_relevance": 0.78
}
```

---

## Phase 3: E1 Memory Retrieval (Days 3-4, 16 hours)

**File**: `e1-memory-retrieval/methodology.md`

### Goal
Compare retrieval strategies and determine E2 hypothesis

### Experimental Design
- **Conditions**: No memory, Temporal (recent 5), Semantic (similar 5)
- **Test set**: 150 problems
- **Runs**: 3 per problem per condition
- **Total**: 1,350 evaluations

### Process

#### Day 3: Setup
1. Generate 1,500 memory problems
2. Generate 150 test problems
3. Run baseline agent on memory set
4. Create SQLite database with embeddings

#### Day 4: Evaluation
1. Run all three conditions
2. Statistical analysis (Friedman test, Wilcoxon post-hoc)
3. Determine E2 hypothesis

### Critical E1 → E2 Decision

| E1 Result | p-value | Effect | E2 Hypothesis | E2 Focus |
|-----------|---------|--------|---------------|-----------|
| Semantic >> Temporal | <0.01 | Large | Amplification | Enhance advantage |
| Semantic ≈ Temporal | >0.10 | None | **Awakening** | Create advantage |
| Semantic < Temporal | <0.05 | Reverse | Correction | Fix bias |
| High variance | Any | CV>0.5 | Stabilization | Reduce variance |

### Key Outputs for E2
- Performance baselines (median attempts per condition)
- Selected hypothesis with rationale
- Token usage baselines

---

## Phase 4: E2 Fine-tuning (Days 5-7, 24 hours)

**File**: `e2-fine-tuning/methodology.md`

### Goal
Test if fine-tuning improves RAG usage based on E1 findings

### Adaptive Design
Design varies based on E1 results. Most likely scenario (Awakening):

**Conditions to Test**:
1. Base + Temporal
2. Base + Semantic
3. Fine-tuned (Summary) + Semantic
4. Fine-tuned (Raw) + Semantic

### Process

#### Day 5: Data Preparation
1. Generate Dataset S (distilled summaries)
2. Generate Dataset R (raw logs)
3. Key difference: S emphasizes *why* semantic helps

#### Day 6: Fine-tuning
```python
# LoRA Configuration
config = {
    "r": 8,
    "alpha": 16,
    "learning_rate": 5e-5,
    "epochs": 3
}
```

#### Day 7: Evaluation & Analysis
1. Test all models on same 150 problems
2. Measure: attempts, tokens, query quality, hypothesis quality
3. Cost-efficiency analysis
4. Statistical tests with Bonferroni correction

### Success Criteria

**Moderate Success** (any one):
- Training time(S) < 65% of (R) AND performance ≥ 95%
- Performance(S) 25% better AND training ≤ 1.5x
- Token usage(S) < 60% AND performance ≥ 90%

**Strong Success** (any one):
- Training time(S) < 50% of (R) AND performance ≥ 85%
- Performance(S) 50% better AND training ≤ 2x

### Key Deliverables
1. **Finding**: "Fine-tuning creates 35% advantage where none existed"
2. **Recommendation**: Deploy model X because Y
3. **Break-even**: N problems to recover training cost

---

## Quick Decision Reference

### When to STOP Completely
1. P1 fails (embeddings don't cluster) → Fundamental flaw
2. P2 fails (retrieval doesn't help) → Core assumption invalid
3. P3 fails (fine-tuning breaks model) → Method unusable

### When to Proceed with Modifications
1. E0 finds no optimal parameters → Use defaults
2. E1 inconclusive → Increase sample size
3. E2 training fails → Adjust hyperparameters

### When to Proceed as Planned
- All prerequisites pass
- E0 finds good parameters
- E1 shows any clear pattern
- E2 hypothesis matches E1 results

---

## Resource Requirements

| Phase | Compute | Storage | Time | Critical? |
|-------|---------|---------|------|-----------|
| Prerequisites | CPU only | 1 GB | 10 hrs | YES - Blocking |
| E0 | CPU only | 2 GB | 8 hrs | NO - Can use defaults |
| E1 | CPU only | 5 GB | 16 hrs | YES - Determines E2 |
| E2 | GPU/Neural Engine | 10 GB | 24 hrs | YES - Main result |

---

## Reproducibility Checklist

Before starting each phase:
- [ ] Set random seeds (42 for generation, 100/200/300 for runs)
- [ ] Record library versions
- [ ] Clear previous outputs
- [ ] Verify compute resources available
- [ ] Check disk space (need 20GB total)

During execution:
- [ ] Save checkpoints every 10 problems
- [ ] Log all hyperparameters
- [ ] Track token usage
- [ ] Monitor for crashes/timeouts
- [ ] Backup database regularly

After completion:
- [ ] Verify all outputs generated
- [ ] Run statistical tests
- [ ] Create visualizations
- [ ] Document any deviations
- [ ] Archive raw data

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Embeddings timeout | Reduce batch size to 10 |
| Database corruption | Restore from checkpoint |
| Model won't converge | Reduce learning rate by 2x |
| Out of memory | Use gradient accumulation |
| Tests inconclusive | Increase runs from 3 to 5 |

---

## Next Steps After E2

Based on results:
1. **If successful** → Prepare deployment pipeline
2. **If mixed** → Run ablation studies
3. **If failed** → Check assumptions, try domain transfer
4. **If surprising** → Investigate mechanism, write paper

---

## Contact & Support

- Hardware issues: Check M2 Pro optimization guides for mlx
- Statistical questions: Refer to methodology.md files
- Implementation details: See code examples in each methodology
- Design rationale: Check archive/ for development history