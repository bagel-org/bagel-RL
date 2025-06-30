#!/usr/bin/env python3
"""
Demo script showcasing all features of the LLM Tool Training Playground.
"""

import json
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")


def print_step(step_num: int, description: str):
    """Print a formatted step."""
    print(f"\n📌 Step {step_num}: {description}")
    print("-" * 40)


def demonstrate_tool_execution():
    """Demonstrate basic tool execution."""
    print_step(1, "Tool Execution Demo")
    
    # Add src to Python path for demo
    demo_script = '''
import sys
sys.path.insert(0, "src")

from tools.executor import ToolExecutor

# Define tools
tools_config = [
    {
        "name": "calculator",
        "description": "Perform calculations",
        "type": "function",
        "function": "calculator",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
    },
    {
        "name": "weather", 
        "description": "Get weather info",
        "type": "function",
        "function": "weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
    }
]

# Initialize executor
executor = ToolExecutor(tools_config)

# Test calculations
print("🧮 Calculator Tests:")
calc_tests = ["2 + 3", "10 * 7", "(15 + 5) / 4", "2 ** 8"]
for expr in calc_tests:
    result = executor.execute_tool("calculator", {"expression": expr})
    print(f"  {expr} = {result.get('result', 'Error')}")

print("\\n🌤️ Weather Tests:")
weather_tests = ["London", "Tokyo", "New York"]
for location in weather_tests:
    result = executor.execute_tool("weather", {"location": location})
    print(f"  {location}: {result.get('temperature', 'N/A')} - {result.get('condition', 'N/A')}")
'''
    
    with open("demo_tools.py", "w") as f:
        f.write(demo_script)
    
    try:
        result = subprocess.run([sys.executable, "demo_tools.py"], 
                              capture_output=True, text=True, cwd=Path.cwd())
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    except Exception as e:
        print(f"Demo failed: {e}")
    finally:
        Path("demo_tools.py").unlink(missing_ok=True)


def demonstrate_data_generation():
    """Demonstrate data generation strategies."""
    print_step(2, "Data Generation Demo")
    
    strategies = [
        ("manual_templates", "Manual templates with paraphrasing"),
        ("teacher_mode", "Teacher mode (Toolformer-style)"),
        ("toolbench", "ToolBench-style synthetic data")
    ]
    
    for strategy, description in strategies:
        print(f"\n🔧 {description}:")
        
        demo_script = f'''
import sys
sys.path.insert(0, "src")

from data.data_generator import DataGenerator

data_config = {{
    "strategy": "{strategy}",
    "max_samples": 5,
    "train_split": 0.8
}}

tools_config = [{{
    "name": "calculator",
    "description": "Perform calculations", 
    "type": "function",
    "function": "calculator",
    "parameters": {{"type": "object", "properties": {{"expression": {{"type": "string"}}}}}}
}}]

generator = DataGenerator(data_config, tools_config)
train_dataset, eval_dataset = generator.prepare_datasets()

print(f"Generated {{len(train_dataset)}} training samples")
print(f"Generated {{len(eval_dataset)}} evaluation samples")

if len(train_dataset) > 0:
    print("\\nSample training data:")
    sample = train_dataset[0]["text"][:200] + "..." if len(train_dataset[0]["text"]) > 200 else train_dataset[0]["text"]
    print(f"  {{sample}}")
'''
        
        with open("demo_data.py", "w") as f:
            f.write(demo_script)
        
        try:
            result = subprocess.run([sys.executable, "demo_data.py"], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            print(result.stdout)
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            Path("demo_data.py").unlink(missing_ok=True)


def show_training_configurations():
    """Show available training configurations."""
    print_step(3, "Training Configurations")
    
    config_files = [
        ("calculator_config.json", "Supervised Fine-tuning with Calculator"),
        ("ppo_config.json", "PPO Training with Rewards"),
        ("teacher_mode_config.json", "Teacher Mode Multi-tool Training")
    ]
    
    for config_file, description in config_files:
        config_path = f"configs/{config_file}"
        if Path(config_path).exists():
            print(f"\n📋 {description}")
            print(f"   Config: {config_file}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"   Method: {config['training']['method']}")
            print(f"   Model: {config['model']['name']}")
            print(f"   Data Strategy: {config['data']['strategy']}")
            print(f"   Tools: {[tool['name'] for tool in config['tools']]}")
            
            print(f"\n   🚀 To train: python train.py --config {config_path}")
            print(f"   🔍 To evaluate: python evaluate.py --model-path outputs/{config_file.replace('.json', '')} --config {config_path}")


def show_evaluation_example():
    """Show evaluation capabilities."""
    print_step(4, "Evaluation Framework")
    
    print("📊 Available Evaluation Metrics:")
    metrics = [
        ("Tool Accuracy", "Correct tool selection rate"),
        ("Format Correctness", "Proper tool call formatting"),
        ("Execution Success", "Successful tool execution rate"), 
        ("Response Quality", "Overall response quality score")
    ]
    
    for metric, description in metrics:
        print(f"  • {metric}: {description}")
    
    print("\n📝 Standard Test Cases Available:")
    try:
        with open("examples/test_cases.json", 'r') as f:
            test_cases = json.load(f)
        
        print(f"  • {len(test_cases)} predefined test cases")
        print("  • Covers: calculator, weather, search tools")
        
        print("\n🔍 Sample Test Case:")
        sample_case = test_cases[0]
        print(f"  Prompt: {sample_case['prompt']}")
        print(f"  Expected Tool: {sample_case['expected_tool']}")
        print(f"  Expected Params: {sample_case['expected_params']}")
        
    except Exception as e:
        print(f"  Error loading test cases: {e}")


def show_next_steps():
    """Show next steps for users."""
    print_step(5, "Getting Started")
    
    steps = [
        "🔧 Setup Environment",
        "   ./setup.sh",
        "",
        "🎯 Run Quick Training Example",
        "   python train.py --config configs/calculator_config.json --output-dir outputs/demo",
        "",
        "🔍 Evaluate Results", 
        "   python evaluate.py --model-path outputs/demo --config configs/calculator_config.json",
        "",
        "🎮 Try Interactive Examples",
        "   python examples/run_examples.py",
        "",
        "🧪 Run Tests",
        "   python validate.py",
        "",
        "📚 Customize Your Tools",
        "   Edit configs/your_config.json",
        "   Add tools in src/tools/executor.py"
    ]
    
    for step in steps:
        print(step)


def main():
    """Main demo function."""
    print_header("LLM Tool Use Training Playground - Complete Demo")
    
    print("🎉 Welcome to the LLM Tool Use Training Playground!")
    print("This demo showcases all the key features of the framework.")
    
    # Run demonstrations
    demonstrate_tool_execution()
    demonstrate_data_generation()
    show_training_configurations()
    show_evaluation_example()
    show_next_steps()
    
    print_header("Demo Complete")
    print("✅ The playground is ready for training LLMs on tool use!")
    print("📖 See README.md for detailed documentation.")
    print("🐛 Found issues? Check validate.py or create an issue.")


if __name__ == "__main__":
    main()
