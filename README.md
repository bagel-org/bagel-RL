# LLM Tool Use Training Playground

A minimal, reproducible framework for training open-source LLMs to invoke external tools using gradient-based reinforcement learning methods.

## Features

- **Multiple Data Sources**: ToolBench integration, teacher mode (Toolformer-style), and manual templating
- **Flexible Training**: Reinforcement learning with PPO, DPO, and supervised fine-tuning
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
python train.py --config configs/sft_calculator_config.json --outdir outputs/calculator_config

# DPO training with teacher mode
python train.py --config configs/dpo_calculator_config.json --outdir outputs/calculator_config
```

3. **Evaluate Trained Model**:
```bash
python evaluate.py --model-path outputs/calculator_config --config configs/calculator_config.json
```

4. **Interactive Examples**:
```bash
python examples/run_examples.py
```

## Project Structure

```
├── configs/           # Configuration files
│   ├── calculator_config.json    # SFT example
│   └── ppo_config.json          # PPO example
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
3. **Teacher Mode**: Self-supervised data generation (Toolformer-style)

## Data Generation Strategies

1. **ToolBench**: Use existing ToolBench datasets (with fallback to synthetic data)
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
python train.py --config configs/calculator_config.json --output-dir outputs/calculator-sft
```


## Evaluation

The framework includes comprehensive evaluation metrics:

- **Tool Accuracy**: Correct tool selection
- **Format Correctness**: Proper tool call formatting
- **Execution Success**: Successful tool execution
- **Response Quality**: Overall response quality

Run evaluation:
```bash
python evaluate.py --model-path outputs/calculator-sft --config configs/calculator_config.json
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
