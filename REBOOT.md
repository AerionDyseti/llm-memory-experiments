# Session Reboot Summary: LLM Memory Experiments

## 🧠 Key Learnings & Insights

### Memory: Experimental Framework Design Validated
- The adaptive E1→E2 design is sound: E1 results determine WHICH hypothesis E2 tests, not WHETHER to test
- "Awakening Theory" insight: Fine-tuning can CREATE advantages that don't exist in base model (not just amplify existing ones)
- This anti-fragile design means any E1 result leads to a valuable E2 experiment

### Memory: Critical Infrastructure Completed
- Complete experimental framework documented and GitHub-ready
- All methodologies written for P1-P3 (prerequisites), E0 (parameter optimization), E1 (retrieval), E2 (fine-tuning)
- Machine capability assessment script created to determine hardware suitability
- Distributed coordination protocols designed for multi-machine execution

### Memory: Git Repository Successfully Initialized
- Private GitHub repo created: https://github.com/AerionDyseti/llm-memory-experiments
- All documentation committed and pushed successfully
- Repository structure clean: no nested directories, organized by experiment phase

### Memory: Session Cleanup Complete
- Old `~/Personal/Development/experiment/` directory consolidated into the GitHub repo
- All redundant files migrated to `/archive/` directory
- Working directory is now single source of truth: `~/Personal/Development/llm-memory-experiments/`

## 🏗️ Decisions Made

### Memory: Technology Stack Confirmed
- **Primary Model**: Phi-3-mini-4k-instruct via Ollama (quantized by default)
- **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers
- **Fine-tuning Framework**: MLX with LoRA (r=8, alpha=16) for Apple Silicon
- **Database**: SQLite with sqlite-vec for vector storage
- **Statistics**: Non-parametric tests (Friedman, Wilcoxon) with Bonferroni correction

### Memory: Experiment Sequencing Strategy
- **Prerequisites (P1-P3)** are blocking gates - must pass before continuing
- **E0** (parameter optimization) with grid search on retrieval hyperparameters
- **E1** (retrieval comparison) tests three strategies: semantic, temporal, hybrid
- **E2** (fine-tuning) uses adaptive hypothesis selection based on E1 results

### Memory: Hardware Distribution Plan
- NOT forcing use of MacBook M2 unless necessary
- Instead: distribute experiments across home network machines using quantized models
- Tier 1 (4-6GB RAM): Prerequisites, embeddings only
- Tier 2 (6-10GB RAM): E0/E1 with 4-bit quantized models
- Tier 3 (10+GB RAM): E2 with full experiments

### Memory: Data Generation Strategy
- 1500 memory problems + 150 test problems (no overlap)
- Four equation types: linear, quadratic, exponential, trigonometric
- Noise: ε ~ N(0, 0.1) for realistic data
- Fixed seeds: 42 for generation, [100, 200, 300] for three independent runs

## 🔧 Working Memory

### Memory: Repository Structure Convention
```
llm-memory-experiments/
├── config.py                    # Central configuration (TO BE CREATED)
├── test_capability.py           # Machine assessment (CREATED)
├── REBOOT.md                    # This file
├── README.md                    # Main experiment guide
├── experiment_flowchart.md      # Execution timeline
├── MACHINE_CAPABILITY_ASSESSMENT.md
├── DISTRIBUTED_COORDINATION.md
├── prerequisites/
│   └── methodology.md
├── e0-parameter-optimization/
│   └── methodology.md
├── e1-memory-retrieval/
│   └── methodology.md
├── e2-fine-tuning/
│   └── methodology.md
├── data/                        # (TO BE CREATED) Generated problems
├── db/                          # (TO BE CREATED) SQLite databases
├── models/                      # (TO BE CREATED) Saved models, checkpoints
├── results/                     # (TO BE CREATED) Experiment outputs
└── archive/                     # Historical design iterations
```

### Memory: Python Module Organization (TO BE CREATED)
- `config.py` - Central configuration management
- `utils/` - Shared utilities:
  - `problem_generation.py` - Create 1500+150 problems
  - `embeddings.py` - Vector embedding functionality
  - `database.py` - SQLite-vec database operations
  - `llm_interface.py` - Ollama integration
  - `statistics.py` - Non-parametric testing (Friedman, Wilcoxon, Cliff's delta)
- `prerequisites/` - P1, P2, P3 validation experiments
- `e0/` - Parameter optimization grid search
- `e1/` - Retrieval strategy comparison
- `e2/` - Fine-tuning with adaptive hypotheses
- `evaluation/` - Analysis and reporting

### Memory: Statistical Analysis Conventions
- Alpha level: 0.05
- Bonferroni-corrected alpha: 0.017 (for 3 tests)
- Non-parametric tests: Friedman test (main), Wilcoxon signed-rank (pairwise)
- Effect size: Cliff's delta (non-parametric alternative to Cohen's d)
- Confidence intervals: Bootstrap with 1000 samples

### Memory: E1→E2 Hypothesis Adaptation Logic
```
IF semantic_advantage >> temporal_advantage
  → E2 Tests AMPLIFICATION (fine-tune to strengthen semantic advantage)
ELIF semantic_advantage ≈ temporal_advantage
  → E2 Tests AWAKENING (fine-tune to CREATE semantic advantage)
ELIF semantic_advantage < temporal_advantage
  → E2 Tests CORRECTION (fine-tune to overcome temporal advantage)
ELIF high_variance
  → E2 Tests STABILIZATION (fine-tune to reduce variance)
```

## 🎯 Session Overview

**Primary Goal/Task**: Build complete experimental infrastructure for testing whether fine-tuning can improve LLM's ability to use RAG memory for mathematical problem-solving.

**Current Status**:
- ✅ Complete experimental methodology documented
- ✅ GitHub repository created and initialized
- ✅ Machine capability assessment script created
- ⏳ **NOW**: Building Python implementation modules (config, utilities, experiments)

**Execution Timeline**: 7 days total
- Prerequisites (P1-P3): ~10 hours
- E0 (parameter optimization): ~8 hours
- E1 (retrieval comparison): ~2 days
- E2 (fine-tuning + evaluation): ~3 days

## ✅ Progress Accomplished

### Files Created/Modified
1. **test_capability.py** - Machine capability assessment script
   - Detects system specs (RAM, CPU, GPU, disk)
   - Tests Ollama installation
   - Tests embedding model loading
   - Runs compute benchmark
   - Generates JSON capability reports
   - Determines which experiments can run

2. **REBOOT.md** (this file)
   - Comprehensive handoff document for session continuity

### Git Commits
1. `8596a5a` - Initial commit: Complete experimental framework
   - Core methodology docs for all phases
   - Machine assessment and coordination guides
   - Experiment flowchart and README

2. `88860b5` - Add machine capability assessment script

3. `0c701eb` - Move MAIN_EXPERIMENT_GUIDE to archive and remove old experiment directory
   - Cleaned up `~/Personal/Development/experiment/` directory
   - Consolidated all files into single repo

### Repository Status
- Clean git status (all committed)
- Private GitHub repo: https://github.com/AerionDyseti/llm-memory-experiments
- Single source of truth: `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/`

## 🔄 Current Context

**Working Directory**: `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/`

**Key Directories**:
- `prerequisites/` - P1-P3 methodology
- `e0-parameter-optimization/` - Grid search methodology
- `e1-memory-retrieval/` - Retrieval comparison methodology
- `e2-fine-tuning/` - Fine-tuning methodology with adaptive hypotheses
- `archive/` - Historical design documents

**Key Files**:
- `README.md` - Main experiment guide and quick reference
- `experiment_flowchart.md` - Detailed execution timeline
- `MACHINE_CAPABILITY_ASSESSMENT.md` - Hardware requirements and assessment guide
- `DISTRIBUTED_COORDINATION.md` - Multi-machine execution protocol
- `test_capability.py` - Capability assessment script (READY TO RUN)

**Dependencies** (to be installed):
- Core: `ollama`, `sentence-transformers`, `numpy`, `scipy`
- Database: `sqlite-vec` (or `sqlite-utils` + `vec` extension)
- Statistics: `scipy.stats` (built-in)
- Fine-tuning: `mlx` (if running on Apple Silicon)
- Utilities: `psutil`, `python-dotenv`

**Environment Setup**:
- Python 3.9+
- Ollama running locally (optional, can use API)
- M2 Pro Mac (preferred for fine-tuning, but experiments designed for multi-machine)

## 🚀 Next Steps

### Immediate (Next Session):
1. **Create core infrastructure modules**:
   - `config.py` - Central configuration management
   - `utils/problem_generation.py` - Generate 1500+150 problems
   - `utils/embeddings.py` - Vector embedding functions
   - `utils/database.py` - SQLite-vec database operations
   - `utils/llm_interface.py` - Ollama integration

2. **Run machine capability assessment**:
   - Execute `test_capability.py` on available machines
   - Collect capability reports
   - Determine experiment distribution

3. **Create prerequisite validators**:
   - `prerequisites/p1_embedding_validity.py` - Validate embedding quality
   - `prerequisites/p2_retrieval_utility.py` - Validate retrieval works
   - `prerequisites/p3_finetuning_preservation.py` - Validate fine-tuning basics

### Medium Term:
4. Run all prerequisites (P1-P3) on best available machine - **BLOCKING GATE**
5. Run E0 parameter optimization to find best retrieval hyperparameters
6. Run E1 retrieval comparison (semantic vs temporal vs hybrid)
7. Analyze E1 results and determine E2 hypothesis
8. Run E2 fine-tuning with selected hypothesis
9. Analyze final results and generate report

### Blockers/Considerations:
- All prerequisites (P1-P3) MUST pass before continuing - no exceptions
- E1 results will determine which E2 hypothesis we test
- Hardware capability will determine which machine runs which experiment
- Need Ollama installed on at least one machine to begin

## 💾 Session Artifacts

### Useful Commands
```bash
# Run capability assessment
cd /Users/kevinwhiteside/Personal/Development/llm-memory-experiments
python test_capability.py

# Push changes to GitHub
git add -A && git commit -m "message" && git push -u origin main

# View git log
git log --oneline -10

# Install dependencies
pip install ollama sentence-transformers numpy scipy psutil
pip install sqlite-vec  # or: pip install sqlite-utils
```

### Configuration Defaults (TO BE IMPLEMENTED)
```python
# Models
LLM_MODEL = "phi3:3.8b-mini-4k-instruct-q4_0"  # 4-bit quantized
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Data generation
TOTAL_MEMORY_PROBLEMS = 1500
TOTAL_TEST_PROBLEMS = 150
PROBLEM_TYPES = ["linear", "quadratic", "exponential", "trigonometric"]
NOISE_STD = 0.1
SEEDS = [42]  # generation; runs use [100, 200, 300]

# Statistics
ALPHA = 0.05
BONFERRONI_CORRECTED_ALPHA = 0.017  # 0.05 / 3
BOOTSTRAP_SAMPLES = 1000

# E1 Retrieval Parameters
TOP_K_VALUES = [1, 3, 5, 10, 15, 20]
SIMILARITY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
TEMPORAL_DECAY_FACTORS = [0.0, 0.5, 1.0]

# E2 Fine-tuning
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 3
LORA_RANK = 8
LORA_ALPHA = 16
```

### GitHub Repository
- URL: https://github.com/AerionDyseti/llm-memory-experiments
- Status: Private repository, ready for implementation
- Branch: main (single branch)
- Latest commit hash: 0c701eb

### Key Documentation Files
- See `README.md` for quick start
- See `experiment_flowchart.md` for detailed timeline
- See `MACHINE_CAPABILITY_ASSESSMENT.md` for hardware requirements
- See `DISTRIBUTED_COORDINATION.md` for multi-machine protocol

## 🧠 Relevant Memories

Session established the following memories during context loading:
- Memory: Experiment Design Evolution
- Memory: Critical Prerequisites
- Memory: Distributed Computing Strategy
- Memory: Experimental Framework
- Memory: Technology Stack
- Memory: Repository Structure
- Memory: Statistical Approach
- Memory: Problem Generation

This session added:
- Memory: Experimental Framework Design Validated
- Memory: Critical Infrastructure Completed
- Memory: Git Repository Successfully Initialized
- Memory: Session Cleanup Complete
- Memory: Technology Stack Confirmed
- Memory: Experiment Sequencing Strategy
- Memory: Hardware Distribution Plan
- Memory: Data Generation Strategy
- Memory: Repository Structure Convention
- Memory: Python Module Organization (TO BE CREATED)
- Memory: Statistical Analysis Conventions
- Memory: E1→E2 Hypothesis Adaptation Logic

---

## REBOOT INSTRUCTIONS FOR NEXT SESSION

**Starting Context:**
You are continuing work on the LLM memory experiments project. The complete experimental framework has been designed and documented. You just finished:
1. Initializing a private GitHub repository
2. Creating the machine capability assessment script
3. Cleaning up old directories and consolidating everything into a single repo

**Your Current Location**: `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/`

**What You're Building**:
A comprehensive experiment to test whether fine-tuning can teach LLMs to better use RAG memory for mathematical problem-solving. The framework has 4 phases:
- Prerequisites (P1-P3): Validation experiments (must pass or stop)
- E0: Parameter optimization via grid search
- E1: Retrieval strategy comparison (semantic vs temporal vs hybrid)
- E2: Fine-tuning with adaptive hypotheses based on E1 results

**Project Structure**:
```
llm-memory-experiments/
├── config.py (TO CREATE)
├── test_capability.py ✓
├── README.md ✓
├── MACHINE_CAPABILITY_ASSESSMENT.md ✓
├── DISTRIBUTED_COORDINATION.md ✓
├── experiment_flowchart.md ✓
├── prerequisites/methodology.md ✓
├── e0-parameter-optimization/methodology.md ✓
├── e1-memory-retrieval/methodology.md ✓
├── e2-fine-tuning/methodology.md ✓
├── archive/ ✓
├── data/ (TO CREATE)
├── db/ (TO CREATE)
├── models/ (TO CREATE)
└── results/ (TO CREATE)
```

**Immediate Next Steps**:
1. Create `config.py` with all configuration defaults
2. Create `utils/` module with:
   - `problem_generation.py` - Create 1500+150 problems
   - `embeddings.py` - Embedding functions using sentence-transformers
   - `database.py` - SQLite-vec operations
   - `llm_interface.py` - Ollama integration
   - `statistics.py` - Non-parametric testing
3. Run `test_capability.py` on available machines
4. Create prerequisite validators (P1, P2, P3)
5. Run prerequisites as blocking gate before E0/E1/E2

**Key Technical Details**:
- Models: Phi-3-mini (quantized), all-MiniLM-L6-v2 embeddings
- Database: SQLite with sqlite-vec vectors
- Fine-tuning: MLX with LoRA on Apple Silicon
- Statistics: Non-parametric (Friedman, Wilcoxon), Bonferroni correction
- Data: 1500 memory + 150 test problems, 4 equation types, noise=0.1

**Important Constraints**:
- Prerequisites (P1-P3) are blocking gates - experiment stops if they fail
- E1 results determine which E2 hypothesis we test (adaptive design)
- Don't force M2 Mac - distribute work across multiple machines with quantized models
- Hardware tiers: Tier 1 (4-6GB), Tier 2 (6-10GB), Tier 3 (10+GB)

**GitHub Repository**:
- Private: https://github.com/AerionDyseti/llm-memory-experiments
- Latest commit: 0c701eb (cleaned up directories)
- Use: `git add -A && git commit -m "message" && git push`

**Files You'll Need to Read First**:
1. `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/README.md`
2. `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/MACHINE_CAPABILITY_ASSESSMENT.md`
3. `/Users/kevinwhiteside/Personal/Development/llm-memory-experiments/prerequisites/methodology.md`

**TODO List for This Session**:
- [ ] Create config.py
- [ ] Create utils/problem_generation.py
- [ ] Create utils/embeddings.py
- [ ] Create utils/database.py
- [ ] Create utils/llm_interface.py
- [ ] Create utils/statistics.py
- [ ] Create prerequisites/p1_embedding_validity.py
- [ ] Create prerequisites/p2_retrieval_utility.py
- [ ] Create prerequisites/p3_finetuning_preservation.py
- [ ] Run test_capability.py on available machines
- [ ] Begin with prerequisite P1

**Session Goals**: Get all core infrastructure modules created and run at least P1 prerequisite validation.

---
**End of Reboot Summary**
