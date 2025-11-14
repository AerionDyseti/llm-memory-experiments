# Distributed Experiment Coordination Guide

## Overview
This guide explains how to coordinate experiments across multiple machines in your home network, share results, and manage the distributed workflow.

---

## Network Architecture

```
┌─────────────────────────────────────────────┐
│           Central Coordinator               │
│         (GitHub Repository)                 │
│     AerionDyseti/llm-memory-experiments    │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┴─────────┬──────────┬──────────┐
    ▼                   ▼          ▼          ▼
┌─────────┐      ┌─────────┐  ┌─────────┐  ┌─────────┐
│Machine 1│      │Machine 2│  │Machine 3│  │Machine N│
│ Weak    │      │ Medium  │  │ Strong  │  │ ...     │
│ (4GB)   │      │ (8GB)   │  │ (16GB)  │  │         │
└─────────┘      └─────────┘  └─────────┘  └─────────┘
```

---

## Setup on Each Machine

### 1. Clone Repository
```bash
# On each machine
git clone https://github.com/AerionDyseti/llm-memory-experiments.git
cd llm-memory-experiments

# Create machine-specific branch
MACHINE_NAME=$(hostname)
git checkout -b machine-$MACHINE_NAME
```

### 2. Install Dependencies Based on Capability
```bash
# Minimal (prerequisites only)
pip install numpy scipy pandas sentence-transformers

# Medium (add Ollama)
curl -fsSL https://ollama.com/install.sh | sh
pip install numpy scipy pandas sentence-transformers ollama

# Full (add fine-tuning)
pip install numpy scipy pandas sentence-transformers ollama mlx mlx-lm
```

### 3. Run Capability Assessment
```bash
python test_capability.py

# This creates capability_report_<hostname>.json
# Commit and push this to your branch
git add capability_report_*.json
git commit -m "Add capability report for $(hostname)"
git push origin machine-$(hostname)
```

---

## Experiment Coordination Protocol

### Phase 1: Capability Collection (Day 0)
```bash
# On coordinator machine, collect all reports
git fetch --all
for branch in $(git branch -r | grep machine-); do
    git checkout $branch
    cp capability_report_*.json reports/
done

# Generate distributed plan
python distribute_workload.py
```

### Phase 2: Task Assignment
Each machine gets a task file:

**tasks/machine1_tasks.json**
```json
{
  "machine": "machine1-hostname",
  "phase": "prerequisites",
  "tasks": [
    {
      "experiment": "P1",
      "config": "prerequisites/p1_config.json",
      "output": "results/p1_results.json"
    }
  ]
}
```

### Phase 3: Execution & Result Collection
```bash
# On each machine
python run_assigned_tasks.py

# This will:
# 1. Read tasks/$(hostname)_tasks.json
# 2. Execute assigned experiments
# 3. Save results to results/
# 4. Commit and push results
```

---

## Shared Storage Strategies

### Option 1: Git-Based (Recommended for Small Data)
```bash
# Each machine commits results
git add results/
git commit -m "Results from $(hostname): $(date)"
git push origin machine-$(hostname)

# Coordinator merges results
git checkout main
for branch in $(git branch -r | grep machine-); do
    git merge $branch --no-ff -m "Merge results from $branch"
done
```

### Option 2: Network File System (NFS)
```bash
# On coordinator, share results directory
sudo apt-get install nfs-kernel-server
echo "/path/to/results *(rw,sync,no_subtree_check)" >> /etc/exports
sudo systemctl restart nfs-kernel-server

# On each machine, mount shared directory
sudo mount coordinator:/path/to/results /local/results
```

### Option 3: Rsync-Based
```bash
# Create sync script on each machine
cat > sync_results.sh << 'EOF'
#!/bin/bash
COORDINATOR="user@coordinator-ip"
rsync -avz results/ $COORDINATOR:~/llm-memory-experiments/results/
EOF

# Run after each experiment phase
./sync_results.sh
```

---

## Execution Scripts

### Master Coordinator Script
```python
#!/usr/bin/env python3
"""
coordinate_experiments.py - Master coordinator for distributed execution
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import paramiko  # For SSH coordination

class ExperimentCoordinator:
    def __init__(self, machines_config: str):
        with open(machines_config) as f:
            self.machines = json.load(f)
        self.results = {}

    def check_machine_status(self, hostname: str) -> bool:
        """Ping machine to check if available."""
        result = subprocess.run(['ping', '-c', '1', hostname],
                              capture_output=True)
        return result.returncode == 0

    def assign_tasks(self) -> Dict:
        """Assign tasks based on machine capabilities."""
        assignments = {}

        # Load capability reports
        for machine in self.machines:
            report_file = f"reports/capability_report_{machine['hostname']}.json"
            with open(report_file) as f:
                capability = json.load(f)

            # Assign based on capability
            if capability['capabilities']['prerequisites']:
                if 'prerequisites' not in assignments:
                    assignments['prerequisites'] = []
                assignments['prerequisites'].append(machine['hostname'])

            if capability['capabilities']['e0_e1']:
                if 'e0_e1' not in assignments:
                    assignments['e0_e1'] = []
                assignments['e0_e1'].append(machine['hostname'])

            if capability['capabilities']['e2_quantized']:
                if 'e2' not in assignments:
                    assignments['e2'] = []
                assignments['e2'].append(machine['hostname'])

        return assignments

    def run_remote_experiment(self, hostname: str, experiment: str) -> bool:
        """Run experiment on remote machine via SSH."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Connect
            ssh.connect(hostname, username=self.machines[hostname]['user'])

            # Run experiment
            cmd = f"cd ~/llm-memory-experiments && python run_experiment.py --experiment {experiment}"
            stdin, stdout, stderr = ssh.exec_command(cmd)

            # Wait for completion
            exit_status = stdout.channel.recv_exit_status()

            if exit_status == 0:
                print(f"✓ {hostname}: {experiment} completed successfully")
                return True
            else:
                print(f"✗ {hostname}: {experiment} failed")
                print(stderr.read().decode())
                return False

        finally:
            ssh.close()

    def collect_results(self) -> None:
        """Collect results from all machines."""
        for machine in self.machines:
            hostname = machine['hostname']

            # Pull latest results
            subprocess.run(['git', 'fetch', f'machine-{hostname}'])
            subprocess.run(['git', 'checkout', f'machine-{hostname}'])

            # Copy results
            result_files = Path('results').glob(f'{hostname}_*.json')
            for result_file in result_files:
                with open(result_file) as f:
                    self.results[str(result_file)] = json.load(f)

        # Save combined results
        with open('results/combined_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def run_phase(self, phase: str) -> None:
        """Run a complete experiment phase."""
        print(f"\n{'='*60}")
        print(f"Running Phase: {phase}")
        print('='*60)

        assignments = self.assign_tasks()

        if phase in assignments:
            machines = assignments[phase]
            print(f"Assigned to: {', '.join(machines)}")

            # Run in parallel
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(machines)) as executor:
                futures = []
                for machine in machines:
                    future = executor.submit(
                        self.run_remote_experiment, machine, phase
                    )
                    futures.append(future)

                # Wait for all to complete
                for future in futures:
                    future.result()

        print(f"Phase {phase} complete!")

    def run_all_experiments(self) -> None:
        """Run all experiment phases in order."""
        phases = ['prerequisites', 'e0', 'e1', 'e2']

        for phase in phases:
            self.run_phase(phase)
            self.collect_results()

            # Check if we should continue
            if not self.check_phase_results(phase):
                print(f"Phase {phase} failed critical checks. Stopping.")
                break

    def check_phase_results(self, phase: str) -> bool:
        """Check if phase results pass critical gates."""
        if phase == 'prerequisites':
            # Check P1, P2, P3 results
            for test in ['p1', 'p2', 'p3']:
                result_file = f'results/{test}_results.json'
                if result_file in self.results:
                    if not self.results[result_file].get('passed', False):
                        return False
        return True


if __name__ == "__main__":
    coordinator = ExperimentCoordinator('machines.json')
    coordinator.run_all_experiments()
```

### Worker Node Script
```python
#!/usr/bin/env python3
"""
run_assigned_tasks.py - Worker script for each machine
"""

import json
import socket
import subprocess
from pathlib import Path
import sys

def get_task_file():
    """Find task file for this machine."""
    hostname = socket.gethostname()
    task_file = Path(f'tasks/{hostname}_tasks.json')

    if not task_file.exists():
        print(f"No tasks found for {hostname}")
        sys.exit(1)

    return task_file

def run_experiment(task):
    """Run a single experiment task."""
    print(f"Running: {task['experiment']}")

    # Map experiment to script
    script_map = {
        'P1': 'prerequisites/run_p1.py',
        'P2': 'prerequisites/run_p2.py',
        'P3': 'prerequisites/run_p3.py',
        'E0': 'e0-parameter-optimization/run_e0.py',
        'E1': 'e1-memory-retrieval/run_e1.py',
        'E2': 'e2-fine-tuning/run_e2.py'
    }

    script = script_map.get(task['experiment'])
    if not script:
        print(f"Unknown experiment: {task['experiment']}")
        return False

    # Run with config
    cmd = [sys.executable, script, '--config', task['config']]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ {task['experiment']} completed")

        # Save output
        output_file = Path(task['output'])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.stdout)

        return True
    else:
        print(f"✗ {task['experiment']} failed")
        print(result.stderr)
        return False

def main():
    # Load tasks
    task_file = get_task_file()
    with open(task_file) as f:
        tasks = json.load(f)

    print(f"Machine: {tasks['machine']}")
    print(f"Phase: {tasks['phase']}")
    print(f"Tasks: {len(tasks['tasks'])}")

    # Run each task
    results = []
    for task in tasks['tasks']:
        success = run_experiment(task)
        results.append({
            'task': task['experiment'],
            'success': success
        })

        if not success and task.get('critical', False):
            print("Critical task failed. Stopping.")
            break

    # Save results summary
    hostname = socket.gethostname()
    summary_file = Path(f'results/{hostname}_summary.json')
    summary_file.parent.mkdir(exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump({
            'machine': hostname,
            'phase': tasks['phase'],
            'results': results
        }, f, indent=2)

    # Commit and push
    subprocess.run(['git', 'add', 'results/'])
    subprocess.run(['git', 'commit', '-m',
                   f'Results from {hostname}: {tasks["phase"]}'])
    subprocess.run(['git', 'push', 'origin', f'machine-{hostname}'])

    print("Tasks complete and results pushed!")

if __name__ == "__main__":
    main()
```

---

## Monitoring Dashboard

Create a simple web dashboard to monitor progress:

```python
#!/usr/bin/env python3
"""
monitor_dashboard.py - Simple web dashboard for monitoring
"""

from flask import Flask, render_template, jsonify
import json
from pathlib import Path
import glob

app = Flask(__name__)

@app.route('/')
def dashboard():
    return '''
    <html>
    <head>
        <title>Experiment Monitor</title>
        <meta http-equiv="refresh" content="10">
    </head>
    <body>
        <h1>Distributed Experiment Status</h1>
        <div id="status"></div>
        <script>
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    let html = '<table border="1">';
                    html += '<tr><th>Machine</th><th>Phase</th><th>Status</th></tr>';
                    for (let machine of data.machines) {
                        html += `<tr>
                            <td>${machine.hostname}</td>
                            <td>${machine.phase}</td>
                            <td>${machine.status}</td>
                        </tr>`;
                    }
                    html += '</table>';
                    document.getElementById('status').innerHTML = html;
                });
        </script>
    </body>
    </html>
    '''

@app.route('/api/status')
def status():
    machines = []

    for summary_file in glob.glob('results/*_summary.json'):
        with open(summary_file) as f:
            data = json.load(f)
            machines.append({
                'hostname': data['machine'],
                'phase': data['phase'],
                'status': 'Complete' if all(r['success']
                         for r in data['results']) else 'Failed'
            })

    return jsonify({'machines': machines})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## Troubleshooting

### Common Issues

1. **Machine goes offline during experiment**
   - Tasks are saved locally and can resume
   - Coordinator will retry or reassign

2. **Different Python versions**
   - Use virtual environments
   - Specify python3.8+ in all scripts

3. **Network connectivity**
   - Ensure all machines can reach coordinator
   - Use static IPs or hostname resolution

4. **Storage filling up**
   - Regular cleanup of old results
   - Compress large files before syncing

### Recovery Procedures

```bash
# If a machine fails mid-experiment
# On coordinator:
python redistribute_failed_tasks.py --failed-machine hostname

# To resume from checkpoint
python run_experiment.py --resume-from checkpoint.json

# To verify result integrity
python verify_results.py --phase e1
```

---

## Best Practices

1. **Always run capability assessment first**
2. **Use quantized models on memory-limited machines**
3. **Implement checkpointing for long-running tasks**
4. **Sync results frequently (every hour)**
5. **Keep logs for debugging**
6. **Use tmux/screen for long-running processes**

---

## Example: 3-Machine Setup

**Machine 1 (Raspberry Pi 4, 4GB)**
- Role: Prerequisites, embeddings
- Tasks: P1, P2, P3

**Machine 2 (Old laptop, 8GB)**
- Role: Parameter search, retrieval
- Tasks: E0, E1 (quantized)

**Machine 3 (Desktop, 16GB)**
- Role: Fine-tuning, final evaluation
- Tasks: E2 (full precision)

Total time: 3-4 days (vs 7 days sequential)