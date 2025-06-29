# LLM Tool Use Training Playground

A minimal, reproducible framework for training open-source LLMs to invoke external tools using gradient-based reinforcement learning methods.

## Features

- **Multiple Data Sources**: ToolBench integration, teacher mode (Toolformer-style), and manual templating
- **Flexible Training**: Reinforcement learning with DPO, and supervised fine-tuning
- **Configuration-Driven**: JSON-based tool definitions and training recipes
- **Reproducible**: Fixed seeds, logging, and evaluation metrics
- **Extensible**: Easy to add new tools and training methods

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

# DPO training with teacher mode
python train.py --config configs/dpo_config.json --outdir outputs/dpo_results
```

3. **Evaluate Trained Model**:
```bash
python evaluate.py --model-path outputs/dpo_results --config configs/dpo_config.json
```

4. **Interactive Examples**:
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

The framework uses JSON configuration files to define:

### Model Configuration
```json
{
  "model": {
    "name": "microsoft/DialoGPT-medium",
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

The playground includes several built-in tools for testing:

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

There are two evaluation criterion for the framework. 
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

## Research Applications

This playground is designed for:

- **Junior Engineers**: Learning RL and tool use
- **Researchers**: Reproducible tool use experiments
- **Prototyping**: Quick experimentation with tool integration



## Contributing

Contributions are welcome! Please see the issues page for areas where help is needed.

## License

MIT License - see LICENSE file for details.
