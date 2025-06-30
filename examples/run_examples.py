#!/usr/bin/env python3
"""
Example usage script demonstrating different training modes.
"""

import subprocess
import sys
from pathlib import Path


def run_training_example(config_name: str):
    """Run a training example."""
    config_path = f"configs/{config_name}"
    
    if not Path(config_path).exists():
        print(f"Configuration file {config_path} not found!")
        return False
    
    print(f"\n🚀 Running training with {config_name}...")
    
    try:
        result = subprocess.run([
            sys.executable, "train.py",
            "--config", config_path,
            "--output-dir", f"outputs/{config_name.replace('.json', '')}"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Training completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    print("🎯 LLM Tool Use Training Playground - Example Usage")
    print("=" * 50)
    
    examples = [
        ("dpo_config.json", "DPO Training with manual templates"),
        ("sft_toolbech_config.json", "Supervised Fine-tuning with Toolbench Data")
        
    ]
    
    for config_file, description in examples:
        print(f"\n📋 Example: {description}")
        print(f"Config: {config_file}")
        
        response = input("Run this example? (y/n): ").lower().strip()
        if response == 'y':
            success = run_training_example(config_file)
            if success:
                print("You can now evaluate the model with:")
                print(f"python evaluate.py --model-path outputs/{config_file.replace('.json', '')} --config configs/{config_file}")
        else:
            print("Skipping...")


if __name__ == "__main__":
    main()
