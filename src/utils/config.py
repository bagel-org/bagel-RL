"""Configuration management utilities."""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _validate_config(self):
        """Validate configuration against schema."""
        schema = {
            "type": "object",
            "required": ["model", "training", "data", "tools"],
            "properties": {
                "model": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "trust_remote_code": {"type": "boolean"},
                        "torch_dtype": {"type": "string"},
                        "device_map": {"type": "string"}
                    }
                },
                "training": {
                    "type": "object",
                    "required": ["method"],
                    "properties": {
                        "method": {"enum": ["sft", "ppo", "dpo", "teacher_mode"]},
                        "num_epochs": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "gradient_accumulation_steps": {"type": "integer", "minimum": 1},
                        "warmup_steps": {"type": "integer", "minimum": 0},
                        "max_length": {"type": "integer", "minimum": 1}
                    }
                },
                "data": {
                    "type": "object",
                    "required": ["strategy"],
                    "properties": {
                        "strategy": {"enum": ["toolbench", "teacher_mode", "manual_templates"]},
                        "max_samples": {"type": "integer", "minimum": 1},
                        "train_split": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "tools": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["name", "description", "parameters"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                }
            }
        }
        
        try:
            jsonschema.validate(self.config, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid configuration: {e.message}")
    
    def get_config(self) -> Dict[str, Any]:
        """Return the loaded configuration."""
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        return self.config.get(section, {})
