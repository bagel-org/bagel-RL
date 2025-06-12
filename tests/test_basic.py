"""Tests for the tool training playground."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools.executor import ToolExecutor
from data.data_generator import DataGenerator
from utils.config import ConfigManager


def test_tool_executor():
    """Test basic tool execution."""
    tools_config = [
        {
            "name": "calculator",
            "description": "Perform calculations",
            "type": "function",
            "function": "calculator",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
        }
    ]
    
    executor = ToolExecutor(tools_config)
    
    # Test calculator
    result = executor.execute_tool("calculator", {"expression": "2 + 3"})
    assert "result" in result
    assert result["result"] == 5
    
    # Test invalid tool
    result = executor.execute_tool("nonexistent", {})
    assert "error" in result


def test_data_generator():
    """Test data generation."""
    data_config = {
        "strategy": "manual_templates",
        "max_samples": 10,
        "train_split": 0.8
    }
    
    tools_config = [
        {
            "name": "calculator",
            "description": "Perform calculations",
            "type": "function",
            "function": "calculator",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
        }
    ]
    
    generator = DataGenerator(data_config, tools_config)
    train_dataset, eval_dataset = generator.prepare_datasets()
    
    assert len(train_dataset) > 0
    assert len(eval_dataset) > 0
    assert "text" in train_dataset[0]


def test_config_validation():
    """Test configuration validation."""
    # Create a temporary config file
    config_data = {
        "model": {"name": "test-model"},
        "training": {"method": "sft"},
        "data": {"strategy": "manual_templates"},
        "tools": [{"name": "test", "description": "test", "parameters": {}}]
    }
    
    config_file = Path("/tmp/test_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    try:
        config_manager = ConfigManager(str(config_file))
        config = config_manager.get_config()
        assert config["model"]["name"] == "test-model"
    finally:
        config_file.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run basic tests
    test_tool_executor()
    test_data_generator()
    test_config_validation()
    print("✅ All tests passed!")
