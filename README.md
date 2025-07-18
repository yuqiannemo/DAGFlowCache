# DAGFlowCache: Multi-Model Cascaded Inference Framework

A lightweight inference framework integrating DAG input structures, KV cache sharing, and CPU offloading for efficient multi-model cascaded reasoning.

## Core Features

- **DAG Input Management**: Supports defining input dependencies via Directed Acyclic Graph (DAG) structures, flexibly handling complex inference workflows.
- **Cascaded Model Inference**: Enables cascaded calls between dual models, where the output of the preceding model serves as input to the subsequent model for progressive reasoning.
- **KV Cache Sharing**: Shares KV caches across multiple models to reduce redundant computations and improve inference efficiency.
- **CPU Offloading Optimization**: Automatically offloads partial computing tasks to the CPU, balancing GPU resource usage.

## Quick Start

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)

### Installation
```bash
git clone https://github.com/yuqiannemo/DAGFlowCache.git
cd DAGFlowCache
pip install -r requirements.txt
```

### Example Usage
```python
from dagflowcache import LLM, SamplingParams, DAGInput

# Initialize model (with shared KV cache)
llm = LLM(
    model_path="your-model-path",
    share_kv_cache=True,
    cpu_offloading=True  # Enable CPU offloading
)

# Define DAG structure input
dag_input = DAGInput()
dag_input.add_node("prompt1", "<|im_start|>user\nintroduce yourself<|im_end|>")
dag_input.add_node("prompt2", "<|im_start|>user\nlist primes under 100<|im_end|>")

# Configure sampling parameters
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

# Run inference
outputs = llm.generate(dag_input, sampling_params)

# Print results
for output in outputs:
    print(f"Completion for node {output.seq_id}:\n{output.text}\n")
```

## Project Structure
```
DAGFlowCache/
├── dagflowcache/          # Core code
│   ├── engine/            # Inference engine (model cascading/cache management)
│   ├── layers/            # Custom layers (attention mechanisms, etc.)
│   ├── models/            # Model definitions
│   └── utils/             # Utility functions (DAG processing/CPU offloading)
├── examples/              # Example scripts
└── requirements.txt       # Dependencies list
```

## Notes
- KV cache sharing currently supports models with the same architecture (e.g., different fine-tuned models from the same base).
- The number of DAG input nodes is recommended to be ≤50 to avoid overly complex dependency calculations.
- CPU offloading strategy can be adjusted via the `offload_ratio` parameter (default: 0.3, meaning 30% of tasks are offloaded to CPU).

## License
This project is open-source under the MIT License. See the LICENSE file for details.
