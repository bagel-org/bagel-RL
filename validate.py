#!/usr/bin/env python3
"""
Quick validation script to check if the playground is set up correctly.
"""

import json
import sys
from pathlib import Path

def check_project_structure():
    """Check if all required files are present."""
    required_files = [
        "requirements.txt",
        "train.py", 
        "evaluate.py",
        "configs/calculator_config.json",
        "configs/ppo_config.json",
        "src/tools/executor.py",
        "src/data/data_generator.py",
        "src/training/trainer.py",
        "examples/test_cases.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def check_configurations():
    """Check if configuration files are valid JSON."""
    config_files = [
        "configs/calculator_config.json",
        "configs/ppo_config.json"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                json.load(f)
            print(f"✅ {config_file} is valid JSON")
        except Exception as e:
            print(f"❌ {config_file} has invalid JSON: {e}")
            return False
    
    return True


def main():
    print("🔍 LLM Tool Use Training Playground - Validation Check")
    print("=" * 55)
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_configurations)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All checks passed! The playground is ready to use.")
        print("\nNext steps:")
        print("1. Run: ./setup.sh (to install dependencies)")
        print("2. Run: python train.py --config configs/calculator_config.json")
        print("3. Or try: python examples/run_examples.py")
    else:
        print("❌ Some checks failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
