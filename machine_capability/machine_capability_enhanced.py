#!/usr/bin/env python3
"""
Enhanced Machine Capability Assessment for LLM Memory Experiments
Comprehensive testing of hardware, software, and performance capabilities.
"""

import sys
import platform
import psutil
import subprocess
import time
import json
import os
import tempfile
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib

try:
    import numpy as np
except ImportError:
    print("Installing numpy for benchmarks...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


class MachineCapabilityTester:
    """Comprehensive machine capability testing framework."""

    def __init__(self):
        self.results = {}
        self.recommendations = []
        self.warnings = []
        self.test_start_time = datetime.now()

    def run_all_tests(self) -> Dict:
        """Run all capability tests and generate report."""
        print("=" * 70)
        print("ENHANCED MACHINE CAPABILITY ASSESSMENT")
        print("=" * 70)
        print(f"Start time: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Phase 1: System Information
        print("Phase 1: Gathering System Information")
        print("-" * 40)
        self.results['system'] = self.get_detailed_system_info()

        # Phase 2: CPU Benchmarks
        print("\nPhase 2: CPU Performance Benchmarks")
        print("-" * 40)
        self.results['cpu'] = self.benchmark_cpu()

        # Phase 3: Memory Tests
        print("\nPhase 3: Memory Capacity and Speed Tests")
        print("-" * 40)
        self.results['memory'] = self.test_memory()

        # Phase 4: Storage I/O
        print("\nPhase 4: Storage I/O Performance")
        print("-" * 40)
        self.results['storage'] = self.benchmark_storage()

        # Phase 5: Software Dependencies
        print("\nPhase 5: Software Dependencies Check")
        print("-" * 40)
        self.results['software'] = self.check_software_dependencies()

        # Phase 6: LLM Specific Tests
        print("\nPhase 6: LLM-Specific Capability Tests")
        print("-" * 40)
        self.results['llm'] = self.test_llm_capabilities()

        # Phase 7: Network Tests (for distributed execution)
        print("\nPhase 7: Network Connectivity Tests")
        print("-" * 40)
        self.results['network'] = self.test_network()

        # Phase 8: Experiment Eligibility Assessment
        print("\nPhase 8: Experiment Eligibility Assessment")
        print("-" * 40)
        self.results['eligibility'] = self.assess_experiment_eligibility()

        # Generate final report
        return self.generate_comprehensive_report()

    def get_detailed_system_info(self) -> Dict:
        """Gather detailed system specifications."""
        info = {
            'hostname': platform.node(),
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
        }

        # CPU Information
        info['cpu'] = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'current_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'min_freq_mhz': psutil.cpu_freq().min if psutil.cpu_freq() else None,
            'max_freq_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else None,
        }

        # Memory Information
        vm = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': round(vm.total / (1024**3), 2),
            'available_gb': round(vm.available / (1024**3), 2),
            'used_gb': round(vm.used / (1024**3), 2),
            'percent_used': vm.percent,
            'swap_total_gb': round(psutil.swap_memory().total / (1024**3), 2),
            'swap_used_gb': round(psutil.swap_memory().used / (1024**3), 2),
        }

        # Storage Information
        disk = psutil.disk_usage('/')
        info['storage'] = {
            'total_gb': round(disk.total / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'percent_used': disk.percent,
        }

        # GPU Detection
        info['gpu'] = self.detect_gpu()

        print(f"✓ System: {info['platform']} {info['architecture']}")
        print(f"✓ CPU: {info['cpu']['physical_cores']} cores ({info['cpu']['logical_cores']} threads)")
        print(f"✓ Memory: {info['memory']['total_gb']}GB total, {info['memory']['available_gb']}GB available")
        print(f"✓ Storage: {info['storage']['free_gb']}GB free of {info['storage']['total_gb']}GB")

        if info['gpu']:
            print(f"✓ GPU: {info['gpu']['name']} ({info['gpu']['type']})")

        return info

    def detect_gpu(self) -> Optional[Dict]:
        """Detect available GPU hardware."""
        gpu_info = None

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_info = {
                    'type': 'nvidia',
                    'name': parts[0],
                    'memory': parts[1] if len(parts) > 1 else 'Unknown',
                    'driver': parts[2] if len(parts) > 2 else 'Unknown',
                }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for AMD GPU (ROCm)
        if not gpu_info:
            try:
                result = subprocess.run(['rocm-smi', '--showproductname'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = {
                        'type': 'amd',
                        'name': 'AMD GPU (ROCm)',
                        'details': result.stdout.strip()
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Check for Apple Silicon
        if not gpu_info and platform.system() == 'Darwin':
            if 'arm' in platform.machine().lower() or 'apple' in platform.processor().lower():
                gpu_info = {
                    'type': 'apple_silicon',
                    'name': 'Apple Silicon (Neural Engine)',
                    'unified_memory': True
                }

        return gpu_info

    def benchmark_cpu(self) -> Dict:
        """Run comprehensive CPU benchmarks."""
        benchmarks = {}

        # 1. Matrix multiplication (floating point)
        print("Running matrix multiplication benchmark...")
        size = 1000
        iterations = 50

        start = time.perf_counter()
        for _ in range(iterations):
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            c = np.dot(a, b)
        matrix_time = time.perf_counter() - start

        benchmarks['matrix_mult'] = {
            'time_seconds': round(matrix_time, 3),
            'gflops': round((2 * size**3 * iterations) / (matrix_time * 1e9), 2)
        }

        # 2. FFT benchmark (for signal processing)
        print("Running FFT benchmark...")
        signal_size = 2**20  # 1M points
        iterations = 100

        signal = np.random.randn(signal_size).astype(np.complex64)
        start = time.perf_counter()
        for _ in range(iterations):
            fft_result = np.fft.fft(signal)
        fft_time = time.perf_counter() - start

        benchmarks['fft'] = {
            'time_seconds': round(fft_time, 3),
            'throughput_msamples_sec': round((signal_size * iterations) / (fft_time * 1e6), 2)
        }

        # 3. Integer operations
        print("Running integer operations benchmark...")
        int_size = 10000000
        int_array = np.random.randint(0, 1000, int_size)

        start = time.perf_counter()
        for _ in range(10):
            sorted_array = np.sort(int_array)
            unique_vals = np.unique(int_array)
        int_time = time.perf_counter() - start

        benchmarks['integer_ops'] = {
            'time_seconds': round(int_time, 3),
            'ops_per_second': round((int_size * 10) / int_time, 0)
        }

        # 4. Multi-threading test
        print("Running multi-threading benchmark...")
        benchmarks['threading'] = self.benchmark_threading()

        # Performance rating
        if benchmarks['matrix_mult']['gflops'] > 50:
            rating = "Excellent"
        elif benchmarks['matrix_mult']['gflops'] > 20:
            rating = "Good"
        elif benchmarks['matrix_mult']['gflops'] > 10:
            rating = "Adequate"
        else:
            rating = "Limited"
            self.warnings.append("CPU performance may limit experiment speed")

        benchmarks['rating'] = rating

        print(f"✓ Matrix multiplication: {benchmarks['matrix_mult']['gflops']} GFLOPS")
        print(f"✓ FFT throughput: {benchmarks['fft']['throughput_msamples_sec']} MS/s")
        print(f"✓ Threading speedup: {benchmarks['threading']['speedup']:.2f}x")
        print(f"✓ CPU Performance Rating: {rating}")

        return benchmarks

    def benchmark_threading(self) -> Dict:
        """Test multi-threading performance."""
        def worker(q, iterations):
            result = 0
            for i in range(iterations):
                result += i ** 2
            q.put(result)

        iterations = 10000000

        # Single-threaded baseline
        start = time.perf_counter()
        result = sum(i**2 for i in range(iterations))
        single_time = time.perf_counter() - start

        # Multi-threaded (use half the logical cores)
        num_threads = max(1, psutil.cpu_count(logical=True) // 2)
        q = queue.Queue()
        threads = []

        start = time.perf_counter()
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(q, iterations // num_threads))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        multi_time = time.perf_counter() - start

        return {
            'single_thread_time': round(single_time, 3),
            'multi_thread_time': round(multi_time, 3),
            'threads_used': num_threads,
            'speedup': round(single_time / multi_time, 2) if multi_time > 0 else 0
        }

    def test_memory(self) -> Dict:
        """Test memory capacity and bandwidth."""
        results = {}

        # 1. Memory allocation test
        print("Testing memory allocation capacity...")
        max_allocation = 0
        test_sizes_gb = [0.5, 1, 2, 4, 8, 16, 32]

        for size_gb in test_sizes_gb:
            try:
                size_bytes = int(size_gb * 1024**3)
                if size_bytes > psutil.virtual_memory().available * 0.8:
                    break

                # Try to allocate
                test_array = np.zeros(size_bytes // 8, dtype=np.float64)
                max_allocation = size_gb
                del test_array
            except (MemoryError, ValueError):
                break

        results['max_allocation_gb'] = max_allocation

        # 2. Memory bandwidth test
        print("Testing memory bandwidth...")
        test_size = min(int(max_allocation * 0.5 * 1024**3 // 8), 500000000)  # Use half of max, cap at ~4GB

        if test_size > 0:
            array = np.random.rand(test_size).astype(np.float64)

            # Read bandwidth
            start = time.perf_counter()
            for _ in range(10):
                sum_val = np.sum(array)
            read_time = time.perf_counter() - start
            read_bandwidth = (test_size * 8 * 10) / (read_time * 1024**3)  # GB/s

            # Write bandwidth
            start = time.perf_counter()
            for _ in range(10):
                array *= 1.00001
            write_time = time.perf_counter() - start
            write_bandwidth = (test_size * 8 * 10) / (write_time * 1024**3)  # GB/s

            results['bandwidth_gb_s'] = {
                'read': round(read_bandwidth, 2),
                'write': round(write_bandwidth, 2)
            }

            del array
        else:
            results['bandwidth_gb_s'] = {'read': 0, 'write': 0}
            self.warnings.append("Unable to test memory bandwidth due to limited memory")

        # 3. Memory pressure simulation
        print("Testing memory under pressure...")
        results['pressure_test'] = self.test_memory_pressure()

        print(f"✓ Max allocation: {results['max_allocation_gb']}GB")
        if 'bandwidth_gb_s' in results:
            print(f"✓ Memory bandwidth: R={results['bandwidth_gb_s']['read']}GB/s, W={results['bandwidth_gb_s']['write']}GB/s")
        print(f"✓ Memory stability: {results['pressure_test']['status']}")

        return results

    def test_memory_pressure(self) -> Dict:
        """Test system behavior under memory pressure."""
        try:
            # Allocate 70% of available memory
            available = psutil.virtual_memory().available
            alloc_size = int(available * 0.7 // 8)

            arrays = []
            for i in range(3):
                arr = np.random.rand(alloc_size // 3).astype(np.float64)
                arrays.append(arr)
                time.sleep(0.1)

            # Perform operations
            result = arrays[0] + arrays[1] * arrays[2]

            # Clean up
            del arrays
            del result

            return {'status': 'stable', 'message': 'System handles memory pressure well'}

        except MemoryError:
            return {'status': 'limited', 'message': 'System struggles under memory pressure'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def benchmark_storage(self) -> Dict:
        """Benchmark storage I/O performance."""
        results = {}

        # Use temp directory for tests
        test_dir = tempfile.gettempdir()
        test_file = os.path.join(test_dir, 'llm_storage_test.bin')

        # Test different block sizes
        block_sizes = {
            '4KB': 4 * 1024,
            '1MB': 1024 * 1024,
            '10MB': 10 * 1024 * 1024
        }

        for size_name, block_size in block_sizes.items():
            print(f"Testing {size_name} block size I/O...")

            # Generate random data
            data = os.urandom(block_size)
            iterations = min(100, max(10, 100 * 1024 * 1024 // block_size))

            # Write test
            start = time.perf_counter()
            with open(test_file, 'wb') as f:
                for _ in range(iterations):
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            write_time = time.perf_counter() - start
            write_speed = (block_size * iterations) / (write_time * 1024 * 1024)  # MB/s

            # Read test
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                for _ in range(iterations):
                    _ = f.read(block_size)
            read_time = time.perf_counter() - start
            read_speed = (block_size * iterations) / (read_time * 1024 * 1024)  # MB/s

            results[size_name] = {
                'read_mb_s': round(read_speed, 2),
                'write_mb_s': round(write_speed, 2),
                'iterations': iterations
            }

            # Clean up
            try:
                os.remove(test_file)
            except:
                pass

        # Random access test
        print("Testing random access I/O...")
        results['random_access'] = self.test_random_access_io()

        # Calculate storage rating
        avg_read = np.mean([v['read_mb_s'] for v in results.values() if isinstance(v, dict) and 'read_mb_s' in v])
        avg_write = np.mean([v['write_mb_s'] for v in results.values() if isinstance(v, dict) and 'write_mb_s' in v])

        if avg_read > 500 and avg_write > 200:
            rating = "SSD (Excellent)"
        elif avg_read > 100 and avg_write > 50:
            rating = "SSD (Good)"
        else:
            rating = "HDD or Limited"
            self.warnings.append("Storage speed may impact experiment performance")

        results['rating'] = rating

        print(f"✓ Sequential read: {avg_read:.0f}MB/s average")
        print(f"✓ Sequential write: {avg_write:.0f}MB/s average")
        print(f"✓ Storage Rating: {rating}")

        return results

    def test_random_access_io(self) -> Dict:
        """Test random access I/O patterns."""
        test_file = os.path.join(tempfile.gettempdir(), 'llm_random_test.bin')
        file_size = 100 * 1024 * 1024  # 100MB

        try:
            # Create test file
            with open(test_file, 'wb') as f:
                f.write(os.urandom(file_size))

            # Random read test
            positions = np.random.randint(0, file_size - 4096, 1000)

            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                for pos in positions:
                    f.seek(pos)
                    _ = f.read(4096)
            random_time = time.perf_counter() - start

            iops = 1000 / random_time

            # Clean up
            os.remove(test_file)

            return {
                'iops': round(iops, 0),
                'latency_ms': round(random_time * 1000 / 1000, 2)
            }

        except Exception as e:
            return {'error': str(e)}

    def check_software_dependencies(self) -> Dict:
        """Check for required and optional software dependencies."""
        dependencies = {}

        # Python packages
        print("Checking Python packages...")
        packages = {
            'numpy': {'required': True, 'import': 'numpy'},
            'scipy': {'required': True, 'import': 'scipy'},
            'pandas': {'required': False, 'import': 'pandas'},
            'sentence_transformers': {'required': True, 'import': 'sentence_transformers'},
            'transformers': {'required': True, 'import': 'transformers'},
            'torch': {'required': False, 'import': 'torch'},
            'tensorflow': {'required': False, 'import': 'tensorflow'},
            'ollama': {'required': True, 'import': 'ollama'},
            'sqlite_vec': {'required': False, 'import': 'sqlite_vec'},
            'mlx': {'required': False, 'import': 'mlx', 'platform': 'Darwin'},
        }

        for name, info in packages.items():
            if 'platform' in info and platform.system() != info['platform']:
                continue

            try:
                __import__(info['import'])
                dependencies[name] = {'installed': True, 'required': info['required']}
                print(f"  ✓ {name}")
            except ImportError:
                dependencies[name] = {'installed': False, 'required': info['required']}
                if info['required']:
                    print(f"  ✗ {name} (REQUIRED)")
                    self.warnings.append(f"Required package '{name}' not installed")
                else:
                    print(f"  ○ {name} (optional)")

        # System tools
        print("\nChecking system tools...")
        tools = {
            'git': 'git --version',
            'ollama': 'ollama --version',
            'docker': 'docker --version',
            'python3': 'python3 --version',
            'pip': 'pip --version',
            'curl': 'curl --version',
            'wget': 'wget --version',
        }

        for tool, command in tools.items():
            try:
                result = subprocess.run(command.split(), capture_output=True,
                                      text=True, timeout=5)
                if result.returncode == 0:
                    dependencies[tool] = {'installed': True, 'version': result.stdout.strip().split('\n')[0]}
                    print(f"  ✓ {tool}")
                else:
                    dependencies[tool] = {'installed': False}
                    print(f"  ✗ {tool}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                dependencies[tool] = {'installed': False}
                print(f"  ✗ {tool}")

        return dependencies

    def test_llm_capabilities(self) -> Dict:
        """Test LLM-specific capabilities."""
        results = {}

        # 1. Test Ollama connection
        print("Testing Ollama service...")
        results['ollama'] = self.test_ollama_service()

        # 2. Test embedding model loading
        print("Testing embedding model capabilities...")
        results['embeddings'] = self.test_embedding_models()

        # 3. Test model inference speed
        if results['ollama'].get('available'):
            print("Testing LLM inference speed...")
            results['inference'] = self.test_llm_inference()
        else:
            results['inference'] = {'status': 'skipped', 'reason': 'Ollama not available'}

        # 4. Estimate fine-tuning capability
        print("Estimating fine-tuning capability...")
        results['finetuning'] = self.estimate_finetuning_capability()

        return results

    def test_ollama_service(self) -> Dict:
        """Test Ollama service availability and models."""
        try:
            # Check if Ollama is running
            result = subprocess.run(['ollama', 'list'], capture_output=True,
                                  text=True, timeout=10)

            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])

                # Check for phi-3-mini
                has_phi3 = any('phi' in model.lower() for model in models)

                return {
                    'available': True,
                    'models': models,
                    'has_phi3': has_phi3,
                    'status': 'ready' if has_phi3 else 'needs_model'
                }
            else:
                return {'available': False, 'error': 'Ollama not responding'}

        except subprocess.TimeoutExpired:
            return {'available': False, 'error': 'Ollama timeout'}
        except FileNotFoundError:
            return {'available': False, 'error': 'Ollama not installed'}
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def test_embedding_models(self) -> Dict:
        """Test embedding model capabilities."""
        try:
            from sentence_transformers import SentenceTransformer

            # Test loading a small model
            model_name = 'all-MiniLM-L6-v2'

            start = time.perf_counter()
            model = SentenceTransformer(model_name)
            load_time = time.perf_counter() - start

            # Test encoding
            test_sentences = ["This is a test"] * 100

            start = time.perf_counter()
            embeddings = model.encode(test_sentences)
            encode_time = time.perf_counter() - start

            # Memory usage
            import sys
            model_size = sys.getsizeof(model) / (1024**2)  # MB

            return {
                'status': 'ready',
                'model': model_name,
                'load_time_s': round(load_time, 2),
                'encode_time_s': round(encode_time, 2),
                'throughput_sents_per_sec': round(100 / encode_time, 0),
                'embedding_dim': embeddings.shape[1]
            }

        except ImportError:
            return {'status': 'not_installed', 'error': 'sentence-transformers not installed'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def test_llm_inference(self) -> Dict:
        """Test LLM inference speed with Ollama."""
        try:
            # Simple inference test
            prompt = "What is 2+2? Answer with just the number."

            start = time.perf_counter()
            result = subprocess.run(
                ['ollama', 'run', 'phi3:mini', prompt],
                capture_output=True, text=True, timeout=30
            )
            inference_time = time.perf_counter() - start

            if result.returncode == 0:
                response_length = len(result.stdout)

                return {
                    'status': 'success',
                    'inference_time_s': round(inference_time, 2),
                    'response_length': response_length,
                    'tokens_per_sec_estimate': round(response_length / inference_time, 0)
                }
            else:
                return {'status': 'failed', 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Inference took too long'}
        except FileNotFoundError:
            return {'status': 'not_available', 'error': 'Ollama not found'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def estimate_finetuning_capability(self) -> Dict:
        """Estimate fine-tuning capabilities based on hardware."""
        mem_gb = psutil.virtual_memory().available / (1024**3)
        gpu = self.results.get('system', {}).get('gpu')

        capabilities = {
            'lora_rank_8': mem_gb >= 4,
            'lora_rank_16': mem_gb >= 8,
            'lora_rank_32': mem_gb >= 16,
            'full_finetuning_3b': mem_gb >= 32,
            'quantized_finetuning': mem_gb >= 6,
        }

        # GPU-specific capabilities
        if gpu:
            if gpu['type'] == 'nvidia':
                capabilities['cuda_available'] = True
                capabilities['recommended_framework'] = 'transformers+peft'
            elif gpu['type'] == 'apple_silicon':
                capabilities['metal_available'] = True
                capabilities['recommended_framework'] = 'mlx'
            else:
                capabilities['recommended_framework'] = 'transformers+peft (CPU)'
        else:
            capabilities['recommended_framework'] = 'transformers+peft (CPU only)'

        # Estimate training time
        if self.results.get('cpu', {}).get('matrix_mult', {}).get('gflops', 0) > 20:
            capabilities['estimated_speed'] = 'fast'
        elif self.results.get('cpu', {}).get('matrix_mult', {}).get('gflops', 0) > 10:
            capabilities['estimated_speed'] = 'moderate'
        else:
            capabilities['estimated_speed'] = 'slow'

        return capabilities

    def test_network(self) -> Dict:
        """Test network connectivity for distributed execution."""
        results = {}

        # Test internet connectivity
        print("Testing internet connectivity...")
        try:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2', '8.8.8.8'],
                capture_output=True, timeout=3
            )
            results['internet'] = result.returncode == 0
        except:
            results['internet'] = False

        # Test GitHub connectivity
        print("Testing GitHub connectivity...")
        try:
            result = subprocess.run(
                ['git', 'ls-remote', 'https://github.com/AerionDyseti/llm-memory-experiments.git'],
                capture_output=True, text=True, timeout=10
            )
            results['github'] = result.returncode == 0
        except:
            results['github'] = False

        # Test local network speed (if possible)
        results['bandwidth'] = 'Not tested'

        print(f"✓ Internet: {'Connected' if results.get('internet') else 'Not connected'}")
        print(f"✓ GitHub: {'Accessible' if results.get('github') else 'Not accessible'}")

        return results

    def assess_experiment_eligibility(self) -> Dict:
        """Determine which experiments this machine can run."""
        eligibility = {
            'prerequisites': {'eligible': False, 'reasons': []},
            'e0_parameter_optimization': {'eligible': False, 'reasons': []},
            'e1_memory_retrieval': {'eligible': False, 'reasons': []},
            'e2_finetuning_full': {'eligible': False, 'reasons': []},
            'e2_finetuning_quantized': {'eligible': False, 'reasons': []},
            'machine_tier': 0,
            'recommended_role': None
        }

        # Get key metrics
        mem_gb = self.results['system']['memory']['available_gb']
        disk_gb = self.results['system']['storage']['free_gb']
        cpu_rating = self.results.get('cpu', {}).get('rating', 'Limited')

        # Prerequisites (P1-P3)
        if mem_gb >= 3 and disk_gb >= 5:
            eligibility['prerequisites']['eligible'] = True
            eligibility['prerequisites']['reasons'].append("Sufficient memory and disk")
        else:
            if mem_gb < 3:
                eligibility['prerequisites']['reasons'].append(f"Need 3GB RAM (have {mem_gb:.1f}GB)")
            if disk_gb < 5:
                eligibility['prerequisites']['reasons'].append(f"Need 5GB disk (have {disk_gb:.1f}GB)")

        # E0 Parameter Optimization
        if mem_gb >= 5 and disk_gb >= 10:
            eligibility['e0_parameter_optimization']['eligible'] = True
            eligibility['e0_parameter_optimization']['reasons'].append("Can run parameter search")
        else:
            if mem_gb < 5:
                eligibility['e0_parameter_optimization']['reasons'].append(f"Need 5GB RAM")
            if disk_gb < 10:
                eligibility['e0_parameter_optimization']['reasons'].append(f"Need 10GB disk")

        # E1 Memory Retrieval
        if mem_gb >= 5 and disk_gb >= 10:
            eligibility['e1_memory_retrieval']['eligible'] = True
            eligibility['e1_memory_retrieval']['reasons'].append("Can run retrieval experiments")
        else:
            eligibility['e1_memory_retrieval']['reasons'].append("Insufficient resources for E1")

        # E2 Fine-tuning
        if mem_gb >= 12 and disk_gb >= 15:
            eligibility['e2_finetuning_full']['eligible'] = True
            eligibility['e2_finetuning_full']['reasons'].append("Can run full precision fine-tuning")
        elif mem_gb >= 6 and disk_gb >= 10:
            eligibility['e2_finetuning_quantized']['eligible'] = True
            eligibility['e2_finetuning_quantized']['reasons'].append("Can run quantized fine-tuning")
        else:
            eligibility['e2_finetuning_full']['reasons'].append(f"Need 12GB RAM for full fine-tuning")
            eligibility['e2_finetuning_quantized']['reasons'].append(f"Need 6GB RAM for quantized")

        # Determine machine tier
        if eligibility['e2_finetuning_full']['eligible']:
            eligibility['machine_tier'] = 3
            eligibility['recommended_role'] = 'Full experiment runner (all phases)'
        elif eligibility['e2_finetuning_quantized']['eligible']:
            eligibility['machine_tier'] = 2
            eligibility['recommended_role'] = 'Quantized experiment runner (all phases with quantization)'
        elif eligibility['e0_parameter_optimization']['eligible']:
            eligibility['machine_tier'] = 2
            eligibility['recommended_role'] = 'Parameter optimization and retrieval experiments'
        elif eligibility['prerequisites']['eligible']:
            eligibility['machine_tier'] = 1
            eligibility['recommended_role'] = 'Prerequisites and data generation'
        else:
            eligibility['machine_tier'] = 0
            eligibility['recommended_role'] = 'Cannot run experiments - consider upgrading'

        # Add performance-based recommendations
        if cpu_rating in ['Limited']:
            self.recommendations.append("Consider using a more powerful machine for faster execution")
        if self.results.get('storage', {}).get('rating', '').startswith('HDD'):
            self.recommendations.append("SSD storage recommended for better I/O performance")

        return eligibility

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive assessment report."""
        test_end_time = datetime.now()
        duration = (test_end_time - self.test_start_time).total_seconds()

        # Build final report
        report = {
            'metadata': {
                'timestamp': self.test_start_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'hostname': platform.node(),
                'report_version': '2.0'
            },
            'results': self.results,
            'recommendations': self.recommendations,
            'warnings': self.warnings
        }

        # Print summary
        print("\n" + "=" * 70)
        print("ASSESSMENT COMPLETE")
        print("=" * 70)

        eligibility = self.results.get('eligibility', {})

        print(f"\n### Machine Tier: {eligibility.get('machine_tier', 0)} ###")
        print(f"Recommended Role: {eligibility.get('recommended_role', 'Unknown')}")

        print("\n### Experiment Eligibility ###")
        for exp_name, exp_data in eligibility.items():
            if isinstance(exp_data, dict) and 'eligible' in exp_data:
                status = "✓" if exp_data['eligible'] else "✗"
                exp_display = exp_name.replace('_', ' ').title()
                print(f"{status} {exp_display}: {exp_data['eligible']}")
                if exp_data['reasons']:
                    for reason in exp_data['reasons']:
                        print(f"    - {reason}")

        if self.recommendations:
            print("\n### Recommendations ###")
            for rec in self.recommendations:
                print(f"• {rec}")

        if self.warnings:
            print("\n### Warnings ###")
            for warning in self.warnings:
                print(f"⚠ {warning}")

        # Save detailed report
        filename = f"capability_report_{platform.node()}_{test_end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✓ Detailed report saved to: {filename}")

        # Create summary file for quick reference
        summary_file = f"capability_summary_{platform.node()}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Machine Capability Summary - {platform.node()}\n")
            f.write(f"Generated: {test_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Machine Tier: {eligibility.get('machine_tier', 0)}\n")
            f.write(f"Role: {eligibility.get('recommended_role', 'Unknown')}\n\n")
            f.write("Eligible Experiments:\n")
            for exp_name, exp_data in eligibility.items():
                if isinstance(exp_data, dict) and exp_data.get('eligible'):
                    f.write(f"  ✓ {exp_name.replace('_', ' ').title()}\n")

        print(f"✓ Summary saved to: {summary_file}")

        return report


def main():
    """Main entry point."""
    print("\nStarting Enhanced Machine Capability Assessment...")
    print("This comprehensive test will take 2-3 minutes.\n")

    tester = MachineCapabilityTester()

    try:
        report = tester.run_all_tests()

        # Return appropriate exit code
        tier = report['results']['eligibility']['machine_tier']
        if tier == 0:
            print("\n❌ This machine cannot run the experiments.")
            print("   Please upgrade RAM or free up disk space.")
            return 1
        elif tier == 1:
            print("\n⚠️  This machine has limited capabilities.")
            print("   Can only run prerequisites and basic tests.")
            return 0
        elif tier == 2:
            print("\n✅ This machine can run most experiments.")
            print("   Some limitations may apply to fine-tuning.")
            return 0
        else:  # tier 3
            print("\n🎉 This machine can run ALL experiments at full capacity!")
            return 0

    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Assessment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())