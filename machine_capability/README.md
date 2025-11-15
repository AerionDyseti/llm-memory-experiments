# Machine Capability Assessment

Standalone tools for assessing machine capabilities for LLM memory experiments.

## Quick Start

```bash
# Navigate to this directory
cd machine_capability

# Create virtual environment and install minimal dependencies
uv sync

# Run the enhanced assessment
uv run python machine_capability_enhanced.py

# Or run the basic assessment (lighter weight)
uv run python test_capability_basic.py
```

## What It Tests

### Enhanced Assessment (`machine_capability_enhanced.py`)
Comprehensive testing including:

1. **System Information**
   - CPU cores and architecture
   - Memory (RAM) availability
   - Storage capacity and type
   - GPU detection (NVIDIA, AMD, Apple Silicon)

2. **CPU Performance**
   - Matrix multiplication (GFLOPS)
   - FFT throughput
   - Integer operations
   - Multi-threading efficiency

3. **Memory Tests**
   - Maximum allocation capacity
   - Memory bandwidth (read/write)
   - Stability under pressure

4. **Storage I/O**
   - Sequential read/write speeds
   - Random access IOPS
   - Storage type detection (SSD vs HDD)

5. **Software Dependencies**
   - Python package availability
   - System tools (git, ollama, docker)
   - LLM service status

6. **LLM Capabilities**
   - Ollama service check
   - Embedding model performance
   - Inference speed estimation
   - Fine-tuning feasibility

7. **Network**
   - Internet connectivity
   - GitHub repository access

8. **Experiment Eligibility**
   - Prerequisites (P1-P3): 3GB RAM, 5GB disk
   - E0-E1: 5GB RAM, 10GB disk
   - E2 (Quantized): 6GB RAM, 10GB disk
   - E2 (Full): 12GB RAM, 15GB disk

### Basic Assessment (`test_capability_basic.py`)
Lighter-weight testing:
- System specs
- Ollama status
- Simple compute benchmark
- Experiment eligibility

## Output Files

After running, you'll find:
- `capability_report_[hostname]_[timestamp].json` - Detailed JSON report
- `capability_summary_[hostname].txt` - Human-readable summary

## Machine Tiers

- **Tier 0**: Cannot run experiments (upgrade needed)
- **Tier 1**: Prerequisites and data generation only
- **Tier 2**: Can run E0/E1 and quantized E2
- **Tier 3**: Can run all experiments at full precision

## Dependencies

Minimal requirements:
- Python 3.9+
- `psutil` - System monitoring
- `numpy` - Benchmarking

Optional (for full testing):
- `sentence-transformers` - Embedding tests
- `ollama` - LLM service integration