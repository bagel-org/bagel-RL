"""Model utilities and custom architectures."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any


class ToolAwareModel(nn.Module):
    """A model wrapper that's aware of tool calling patterns."""
    
    def __init__(self, base_model, config: Dict[str, Any]):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Tool detection head
        self.tool_detector = nn.Linear(
            base_model.config.hidden_size, 
            len(config.get("tools", []))
        )
        
        # Tool confidence head
        self.confidence_head = nn.Linear(
            base_model.config.hidden_size,
            1
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with tool awareness."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Get last hidden state for tool detection
        last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
        
        if last_hidden_state is not None:
            # Tool detection logits
            tool_logits = self.tool_detector(last_hidden_state[:, -1, :])
            
            # Tool confidence
            confidence = torch.sigmoid(self.confidence_head(last_hidden_state[:, -1, :]))
            
            # Add to outputs
            outputs.tool_logits = tool_logits
            outputs.tool_confidence = confidence
        
        return outputs


class RewardModel(nn.Module):
    """Reward model for tool use success."""
    
    def __init__(self, base_model_name: str, num_tools: int):
        super().__init__()
        
        # Load base model
        config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Tool-specific reward heads
        self.tool_reward_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, 1) for _ in range(num_tools)
        ])
    
    def forward(self, input_ids, attention_mask=None, tool_id=None):
        """Forward pass for reward calculation."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled representation
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # General reward
        reward = self.reward_head(pooled_output)
        
        # Tool-specific reward if tool_id provided
        tool_reward = None
        if tool_id is not None and 0 <= tool_id < len(self.tool_reward_heads):
            tool_reward = self.tool_reward_heads[tool_id](pooled_output)
        
        return {
            "reward": reward,
            "tool_reward": tool_reward,
            "pooled_output": pooled_output
        }
