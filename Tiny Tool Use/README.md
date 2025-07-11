# Tiny Tool Use

An intentionally-tiny yet production-ready open-source library for fine-tuning Large Language Models (LLMs) to make robust, auditable tool calls.

## Features

- **Configuration-only workflows** – every experiment, tool schema, and hyper-parameter lives in a JSON file so results travel cleanly between repos.
- **Interchangeable optimisers** – swap Supervised Fine-Tuning, Direct Preference Optimisation (DPO), or synthetic teacher signals with a single config flag.
- **First-class evaluation support** – TensorBoard dashboards and ready-made Berkeley Function Calling Leaderboard scripts.
- **Dataset flexibility** – plug in real data, generate synthetic traces, or compose both without touching core code.

## Quick Start

1. **Setup Environment**:
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

2. **Run Basic Training**:
```bash
# Supervised fine-tuning with manual templates
python train.py --config configs/sft_toolbench_config.json --outdir outputs/toolbench_results

# DPO training with manual templates
python train.py --config configs/dpo_config.json --outdir outputs/dpo_results
```

3. **Merging LORA Adapters**

If you are using lora adapters to finetune the model, you can merge the lora adapters once your training finishes, using the followig 

```bash
python save_merge_model.py --base_model BASE-MODEL-NAME --adapter_path PATH/TO/SAVED/ADAPTER --output_dir PATH/TO/MERGED/MODEL
```


4. **Evaluate Trained Model**:
```bash
python evaluate.py --model-path PATH/TO/FINAL/SAVED/MODEL --config PATH/TO/TRAINING/CONFIGURATION/JSON/FILE
```

5. **Interactive Examples**:
```bash
python examples/run_examples.py
```

## Project Structure

```
├── configs/           # Configuration files
│   ├── sft_toolbench_config.json    # SFT example with real toolbench data
│   └── dpo_config.json          # DPO example
├── src/              # Source code
│   ├── data/         # Data generation and loading
│   ├── models/       # Model definitions
│   ├── training/     # Training loops and algorithms
│   ├── tools/        # Tool handling
│   └── utils/        # Utilities and evaluation
├── examples/         # Example configurations and tools
│   ├── run_examples.py          # Interactive examples
│   └── test_cases.json         # Standard test cases
├── tests/           # Unit tests
├── train.py         # Main training script
├── evaluate.py      # Evaluation script
└── setup.sh        # Setup script
```

## Configuration

The library uses JSON configuration files to define:

### Model Configuration
```json
{
  "model": {
    "name": "Qwen/Qwen3-0.6B",
    "trust_remote_code": false,
    "torch_dtype": "float16",
    "device_map": "auto"
  }
}
```

### Training Configuration
```json
{
  "training": {
    "method": "sft",  // "sft", "dpo", "teacher_mode"
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "use_lora": true
  }
}
```

### Tool Definitions
If you want to train your model on custom tools with sythetic data generated for custom tools, you can 
define the tools as well as the dataset.
```json
{
  "tools": [
    {
      "name": "calculator",
      "description": "Perform mathematical calculations",
      "type": "function",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate"
          }
        },
        "required": ["expression"]
      }
    }
  ]
}
```

## Supported Training Methods

1. **Supervised Fine-tuning (SFT)**: Standard next-token prediction on tool-augmented conversations
2. **DPO**: Direct Preference Optimization using preference pairs
3. **Teacher Mode**: Self-supervised data generation (Toolformer-style) with synthetic data

## Data Generation Strategies

1. **ToolBench**: Use both real and synthetic tool bench datasets
2. **Teacher Mode**: LLM generates its own tool-augmented examples
3. **Manual Templates**: Bootstrap from canonical examples with paraphrasing

## Built-in Tools

The library includes several built-in tools for testing:

- **Calculator**: Basic arithmetic operations
- **Weather**: Mock weather API
- **Search**: Mock search functionality

You can easily add custom tools by extending the `ToolExecutor` class.

## Training Examples

### Example 1: Calculator with SFT
```bash
python train.py --config configs/sft_toolbench_config.json --output-dir outputs/sft_toolbench
```


## Evaluation

There are two evaluation criteria for the library. 
1. Berkeley Function Calling Leaderboard evaluation

2. Other comprehensive evaluation metrics:

  - **Tool Accuracy**: Correct tool selection
  - **Format Correctness**: Proper tool call formatting
  - **Execution Success**: Successful tool execution
  - **Response Quality**: Overall response quality


1. To run the evaluation of a trained model on Berkeley Function Calling Leaderboard (BFCL) use the following instruction. 

```bash
# In your shell environment
export BFCL_PROJECT_ROOT=/path/to/your/desired/project/directory
```


Run evaluation on BFCL using finetuned `Qwen3-0.6B` model 
```bash
bfcl generate --model Qwen/Qwen3-0.6B-FC --local-model-path PATH_TO_FINETUNED_MODEL --test-category simple,parallel,multiple,multuturn
```
This will create a directory `result/` and the generated `json` files within this directory. 
Once the model response are generated with BFCL run the following command to evaluate the performance of the trained model

```bash
bfcl evaluate --model Qwen/Qwen3-0.6B-FC --test-category simple,parallel,multiple,multuturn
```

2. 
Run comprehensive evaluation with `Qwen3-0.6B` finetuned using DPO:
```bash
python evaluate.py --model-path PATH/TO/FINETUNED/MODEL --config PATH/TO/CONFIG/FILE
```

For example if training is performed with `dpo_config.json`, then for comprehensive evaluation

```bash
python evaluate.py --model-path PATH/TO/FINETUNED/MODEL --config dpo_config.json
```


## Customization

### Adding New Tools

1. Define tool schema in configuration:
```json
{
  "name": "my_tool",
  "description": "My custom tool",
  "type": "function",
  "parameters": {...}
}
```

2. Implement tool function in `src/tools/executor.py`:
```python
def _my_tool(self, parameters):
    # Implementation here
    return {"result": "success"}
```

### Custom Training Methods

Extend the `ToolTrainer` class in `src/training/trainer.py` to add new training algorithms.

### Custom Data Sources

Implement new data generation strategies in `src/data/data_generator.py`.

## Potential Applications

This library can serve as a foundation for a variety of tool-use research problems and production scenarios, including:

- **Robotics control** – grounding language instructions into low-level robot actions through tool calls.
- **Autonomous agents** – building multi-step assistants that plan, call, and combine external APIs.
- **Workflow automation** – integrating structured tool calls into data-engineering or MLOps pipelines.
- **Information retrieval** – augmenting LLM responses with live search or specialized knowledge bases.
- **Education & tutoring systems** – teaching models to execute calculators, solvers, or simulators on demand.

## Contributing

Contributions are welcome! Please see the issues page for areas where help is needed.

## License

MIT License - see LICENSE file for details.
