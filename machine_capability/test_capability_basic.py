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
