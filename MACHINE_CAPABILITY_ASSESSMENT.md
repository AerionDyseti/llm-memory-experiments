# Machine Capability Assessment Guide

## Overview
This guide helps you determine if a machine can run the LLM memory experiments and what adjustments may be needed for resource-constrained systems.

---

## Minimum Requirements by Experiment Phase

### Prerequisites (P1-P3)
**Absolute Minimum:**
- RAM: 4GB
- Storage: 5GB free
- CPU: Any x86_64 or ARM processor
- GPU: Not required

**Recommended:**
- RAM: 8GB
- Storage: 10GB free
- CPU: 4+ cores

### E0-E1 (Parameter Optimization & Retrieval)
**Absolute Minimum:**
- RAM: 6GB
- Storage: 10GB free
- CPU: 2+ cores
- GPU: Not required

**Recommended:**
- RAM: 8GB
- Storage: 20GB free
- CPU: 4+ cores

### E2 (Fine-tuning)
**Absolute Minimum:**
- RAM: 8GB (with quantization)
- Storage: 15GB free
- CPU: 4+ cores
- GPU: Helpful but not required with quantization

**Recommended:**
- RAM: 16GB
- Storage: 30GB free
- CPU: 8+ cores
- GPU: Any CUDA-capable or Apple Silicon

---

## Quick Capability Test Script

Save this as `test_capability.py` and run on each machine:

```python
#!/usr/bin/env python3
"""
Machine Capability Assessment for LLM Memory Experiments
Run this script to determine if your machine can handle the experiments.
"""

import sys
import platform
import psutil
import subprocess
import time
import json
from pathlib import Path

def get_system_info():
    """Gather system specifications."""
    info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'architecture': platform.machine(),
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
    }

    # Check for GPU
    try:
        # NVIDIA GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                               '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            info['gpu'] = result.stdout.strip()
            info['gpu_type'] = 'nvidia'
    except:
        info['gpu'] = None

    # Check for Apple Silicon
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        info['gpu_type'] = 'apple_silicon'
        info['gpu'] = 'Apple Silicon (Neural Engine available)'

    return info

def test_ollama():
    """Test if Ollama is installed and working."""
    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout
    except:
        pass
    return False, None

def test_model_loading():
    """Test loading a small model to check memory."""
    test_script = '''
import time
import psutil

# Record baseline memory
baseline = psutil.virtual_memory().used / (1024**3)
print(f"Baseline memory: {baseline:.2f} GB")

try:
    # Try to import and use a small model
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test encoding
    test_sentences = ["Test sentence"] * 100
    embeddings = model.encode(test_sentences)

    peak = psutil.virtual_memory().used / (1024**3)
    print(f"Peak memory: {peak:.2f} GB")
    print(f"Memory used: {peak - baseline:.2f} GB")

    print("SUCCESS: Can load embedding model")

except ImportError:
    print("SKIP: sentence-transformers not installed")
except MemoryError:
    print("FAIL: Insufficient memory for embedding model")
except Exception as e:
    print(f"FAIL: {e}")
'''

    try:
        result = subprocess.run([sys.executable, '-c', test_script],
                              capture_output=True, text=True, timeout=30)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Model loading took too long (>30s)"
    except Exception as e:
        return f"ERROR: {e}"

def benchmark_compute():
    """Run a simple compute benchmark."""
    import numpy as np

    print("Running compute benchmark...")
    size = 1000

    # Matrix multiplication benchmark
    start = time.time()
    for _ in range(100):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
    duration = time.time() - start

    return {
        'matrix_mult_time': round(duration, 2),
        'operations_per_second': round(100 / duration, 2)
    }

def determine_capability(info, benchmark):
    """Determine what experiments this machine can run."""
    capabilities = {
        'prerequisites': False,
        'e0_e1': False,
        'e2_full': False,
        'e2_quantized': False,
        'model_recommendations': []
    }

    # Prerequisites (P1-P3)
    if info['ram_available_gb'] >= 3 and info['disk_free_gb'] >= 5:
        capabilities['prerequisites'] = True

    # E0-E1
    if info['ram_available_gb'] >= 5 and info['disk_free_gb'] >= 10:
        capabilities['e0_e1'] = True

    # E2 (Fine-tuning)
    if info['ram_available_gb'] >= 12 and info['disk_free_gb'] >= 15:
        capabilities['e2_full'] = True
        capabilities['model_recommendations'].append('phi-3-mini (full precision)')
    elif info['ram_available_gb'] >= 6 and info['disk_free_gb'] >= 10:
        capabilities['e2_quantized'] = True
        capabilities['model_recommendations'].append('phi-3-mini (4-bit quantized)')

    # Adjust for compute speed
    if benchmark['matrix_mult_time'] > 10:
        capabilities['notes'] = "Slow CPU - expect longer runtimes"

    # GPU recommendations
    if info.get('gpu_type') == 'nvidia':
        capabilities['model_recommendations'].append('Consider using GGML/GGUF formats')
    elif info.get('gpu_type') == 'apple_silicon':
        capabilities['model_recommendations'].append('Use MLX for optimal performance')

    return capabilities

def generate_report(info, ollama_status, model_test, benchmark, capabilities):
    """Generate a capability report."""
    print("\n" + "="*60)
    print("MACHINE CAPABILITY ASSESSMENT REPORT")
    print("="*60)

    print("\n### System Specifications ###")
    print(f"Platform: {info['platform']} {info['architecture']}")
    print(f"CPU: {info['cpu_count_physical']} cores ({info['cpu_count']} threads)")
    print(f"RAM: {info['ram_gb']} GB total, {info['ram_available_gb']} GB available")
    print(f"Disk: {info['disk_free_gb']} GB free")
    print(f"GPU: {info.get('gpu', 'None detected')}")

    print("\n### Software Status ###")
    print(f"Ollama: {'✓ Installed' if ollama_status else '✗ Not found'}")

    print("\n### Performance ###")
    print(f"Matrix operations: {benchmark['operations_per_second']} ops/sec")
    print(f"Benchmark time: {benchmark['matrix_mult_time']}s")

    print("\n### Experiment Capabilities ###")
    print(f"Prerequisites (P1-P3): {'✓ Yes' if capabilities['prerequisites'] else '✗ No'}")
    print(f"E0-E1 (Optimization/Retrieval): {'✓ Yes' if capabilities['e0_e1'] else '✗ No'}")
    print(f"E2 (Fine-tuning - Full): {'✓ Yes' if capabilities['e2_full'] else '✗ No'}")
    print(f"E2 (Fine-tuning - Quantized): {'✓ Yes' if capabilities['e2_quantized'] else '✗ No'}")

    if capabilities['model_recommendations']:
        print("\n### Recommendations ###")
        for rec in capabilities['model_recommendations']:
            print(f"- {rec}")

    if capabilities.get('notes'):
        print(f"\nNotes: {capabilities['notes']}")

    # Save report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': platform.node(),
        'system': info,
        'benchmark': benchmark,
        'capabilities': capabilities
    }

    filename = f"capability_report_{platform.node()}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {filename}")

    return capabilities

def main():
    print("Starting machine capability assessment...")
    print("This will take about 30-60 seconds.\n")

    # Gather system info
    print("1. Checking system specifications...")
    info = get_system_info()

    # Test Ollama
    print("2. Checking for Ollama...")
    ollama_status, ollama_models = test_ollama()

    # Test model loading
    print("3. Testing model loading capabilities...")
    model_test = test_model_loading()
    print(model_test)

    # Run benchmark
    print("\n4. Running compute benchmark...")
    benchmark = benchmark_compute()

    # Determine capabilities
    print("5. Analyzing capabilities...")
    capabilities = determine_capability(info, benchmark)

    # Generate report
    report = generate_report(info, ollama_status, model_test,
                           benchmark, capabilities)

    # Return code based on capability
    if not capabilities['prerequisites']:
        print("\n⚠️  This machine cannot run the experiments.")
        return 1
    elif capabilities['e2_full']:
        print("\n✅ This machine can run ALL experiments at full precision!")
        return 0
    elif capabilities['e2_quantized']:
        print("\n✅ This machine can run all experiments with quantization.")
        return 0
    elif capabilities['e0_e1']:
        print("\n⚠️  This machine can only run Prerequisites, E0, and E1.")
        return 0
    else:
        print("\n⚠️  This machine can only run Prerequisites.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## Model Quantization Options

For memory-constrained machines, use quantized models:

### Phi-3 Quantization Levels

| Quantization | RAM Required | Quality Impact | Use Case |
|--------------|--------------|----------------|----------|
| Full (FP16) | 8-12 GB | None | Best quality |
| 8-bit | 4-6 GB | Minimal | Good balance |
| 4-bit | 2-3 GB | Small | Memory constrained |
| 3-bit | 1.5-2 GB | Noticeable | Very limited RAM |

### Installing Quantized Models

#### Option 1: Ollama (Recommended for CPU)
```bash
# Install 4-bit quantized version
ollama pull phi3:3.8b-mini-4k-instruct-q4_0

# For very constrained systems (3-bit)
ollama pull phi3:3.8b-mini-4k-instruct-q3_K_M
```

#### Option 2: GGUF Format (CPU + Limited GPU)
```python
# Install llama-cpp-python
pip install llama-cpp-python

# Download quantized model
wget https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF/resolve/main/phi-3-mini-4k-instruct.Q4_K_M.gguf
```

#### Option 3: MLX (Apple Silicon only)
```python
# Install MLX
pip install mlx mlx-lm

# Use 4-bit quantization
from mlx_lm import load, generate
model, tokenizer = load("microsoft/Phi-3-mini-4k-instruct",
                        quantize={"group_size": 64, "bits": 4})
```

---

## Distributed Execution Strategy

### Phase Distribution

Since you have multiple machines, you can parallelize:

#### Tier 1: Weak Machines (4-6GB RAM)
- Run Prerequisites (P1-P3)
- Run subset of E0 parameter search
- Generate training data

#### Tier 2: Medium Machines (6-10GB RAM)
- Run E0 parameter optimization
- Run E1 retrieval experiments
- Use quantized models

#### Tier 3: Stronger Machines (10+ GB RAM)
- Run E2 fine-tuning
- Run full evaluations
- Generate final results

### Workload Distribution Script

```python
"""
distribute_workload.py - Assigns experiments to machines based on capability
"""

import json
import glob
from pathlib import Path

def load_capability_reports():
    """Load all capability reports."""
    reports = []
    for report_file in glob.glob("capability_report_*.json"):
        with open(report_file) as f:
            reports.append(json.load(f))
    return reports

def assign_workload(reports):
    """Assign experiments based on machine capabilities."""
    assignments = {
        'tier1': [],  # Weak machines
        'tier2': [],  # Medium machines
        'tier3': []   # Strong machines
    }

    for report in reports:
        hostname = report['hostname']
        caps = report['capabilities']
        ram = report['system']['ram_available_gb']

        if caps['e2_full']:
            assignments['tier3'].append({
                'hostname': hostname,
                'tasks': ['e2_finetuning', 'e2_evaluation'],
                'model': 'phi3-full'
            })
        elif caps['e2_quantized']:
            assignments['tier2'].append({
                'hostname': hostname,
                'tasks': ['e1_retrieval', 'e0_optimization'],
                'model': 'phi3-q4'
            })
        elif caps['e0_e1']:
            assignments['tier2'].append({
                'hostname': hostname,
                'tasks': ['e0_subset', 'e1_subset'],
                'model': 'phi3-q4'
            })
        elif caps['prerequisites']:
            assignments['tier1'].append({
                'hostname': hostname,
                'tasks': ['prerequisites', 'data_generation'],
                'model': 'embeddings_only'
            })

    return assignments

def generate_execution_plan(assignments):
    """Create execution plan for distributed run."""
    plan = {
        'phase1': [],  # Parallel prerequisites
        'phase2': [],  # Parallel E0/E1
        'phase3': []   # E2 (may need single machine)
    }

    # Phase 1: Prerequisites on all tier1 machines
    for machine in assignments['tier1']:
        plan['phase1'].append({
            'machine': machine['hostname'],
            'experiment': 'prerequisites',
            'parallel': True
        })

    # Phase 2: E0/E1 distributed
    for machine in assignments['tier2']:
        plan['phase2'].append({
            'machine': machine['hostname'],
            'experiment': 'e0_e1_subset',
            'parallel': True
        })

    # Phase 3: E2 on best machine
    if assignments['tier3']:
        plan['phase3'].append({
            'machine': assignments['tier3'][0]['hostname'],
            'experiment': 'e2_full',
            'parallel': False
        })
    elif assignments['tier2']:
        # Fall back to quantized on best tier2
        plan['phase3'].append({
            'machine': assignments['tier2'][0]['hostname'],
            'experiment': 'e2_quantized',
            'parallel': False
        })

    return plan

# Usage
if __name__ == "__main__":
    reports = load_capability_reports()
    assignments = assign_workload(reports)
    plan = generate_execution_plan(assignments)

    print("Execution Plan:")
    print(json.dumps(plan, indent=2))

    with open("distributed_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
```

---

## Memory Optimization Techniques

### 1. Batch Size Adjustment
```python
# Reduce batch size for limited memory
if available_ram < 8:
    batch_size = 1
elif available_ram < 16:
    batch_size = 2
else:
    batch_size = 4
```

### 2. Gradient Accumulation
```python
# Simulate larger batches on limited memory
gradient_accumulation_steps = desired_batch_size // actual_batch_size
```

### 3. Model Loading Strategy
```python
# Load models on demand, not all at once
def load_model_on_demand(model_name):
    model = load_model(model_name)
    result = model.generate(...)
    del model  # Free memory immediately
    return result
```

### 4. Disk Caching
```python
# Use disk for intermediate results
import joblib

# Save memory by caching to disk
joblib.dump(large_object, 'cache/temp.pkl')
del large_object

# Load when needed
large_object = joblib.load('cache/temp.pkl')
```

---

## Recommended Setup by Machine Type

### Raspberry Pi / SBC (2-4GB RAM)
- Role: Data generation, embedding computation
- Model: None (embeddings only)
- Experiments: Prerequisites only

### Old Desktop/Laptop (4-8GB RAM)
- Role: Parameter search, retrieval testing
- Model: Phi-3 Q4 quantized
- Experiments: P1-P3, E0 (subset), E1 (subset)

### Modern Laptop (8-16GB RAM)
- Role: Full retrieval, quantized fine-tuning
- Model: Phi-3 Q4 or Q8
- Experiments: All except full E2

### Desktop with GPU (16+ GB RAM)
- Role: Full experiments, fine-tuning
- Model: Phi-3 full precision
- Experiments: All experiments

---

## Quick Decision Tree

```
1. Run test_capability.py
   ↓
2. RAM < 4GB?
   → Only prerequisites with embeddings

3. RAM 4-8GB?
   → Prerequisites + E0/E1 with Q4 models

4. RAM 8-16GB?
   → All experiments with Q4/Q8 models

5. RAM > 16GB?
   → All experiments at full precision
```

---

## Next Steps

1. Run `test_capability.py` on each available machine
2. Collect all `capability_report_*.json` files
3. Run `distribute_workload.py` to create execution plan
4. Install appropriate quantized models on each machine
5. Begin distributed execution following the plan

This approach will let you leverage all available hardware efficiently!