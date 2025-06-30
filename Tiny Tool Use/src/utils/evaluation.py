"""Evaluation utilities for tool use models."""

import json
import logging
import re
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from ..tools.executor import ToolExecutor


logger = logging.getLogger(__name__)


class ToolUseEvaluator:
    """Evaluates model performance on tool use tasks."""
    
    def __init__(self, model_path: str, tools_config: List[Dict[str, Any]]):
        self.model_path = model_path
        self.tools_config = tools_config
        self.tool_executor = ToolExecutor(tools_config)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_dataset(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        results = {
            "tool_accuracy": 0.0,
            "format_correctness": 0.0,
            "execution_success": 0.0,
            "response_quality": 0.0
        }
        
        total_samples = len(eval_dataset)
        correct_tools = 0
        correct_format = 0
        successful_executions = 0
        
        for example in eval_dataset:
            # Generate response
            response = self._generate_response(example["text"])
            
            # Extract ground truth tool
            gt_tool = example.get("tool_name", "")
            
            # Evaluate tool accuracy
            predicted_tool = self._extract_tool_name(response)
            if predicted_tool == gt_tool:
                correct_tools += 1
            
            # Evaluate format correctness
            if self._check_format_correctness(response):
                correct_format += 1
            
            # Evaluate execution success
            if self._check_execution_success(response):
                successful_executions += 1
        
        results["tool_accuracy"] = correct_tools / total_samples
        results["format_correctness"] = correct_format / total_samples
        results["execution_success"] = successful_executions / total_samples
        results["response_quality"] = (results["tool_accuracy"] + results["format_correctness"]) / 2
        
        return results
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response for a given prompt."""
        # Extract just the human part for generation
        if "Human:" in prompt and "Assistant:" in prompt:
            human_part = prompt.split("Assistant:")[0] + "Assistant:"
        else:
            human_part = prompt
        
        # Encode input and move to the same device as the model
        inputs = self.tokenizer.encode(human_part, return_tensors="pt")
        device = self.model.device  # Get the model's device
        inputs = inputs.to(device)  # Move input tensor to the same device
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
       
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the generated part
        if human_part in response:
            response = response.replace(human_part, "").strip()
        
        return response
    
    def _extract_tool_name(self, response: str) -> str:
        """Extract tool name from response."""
        # Look for tool call pattern
        tool_call_pattern = r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if matches:
            try:
                tool_data = json.loads(matches[0])
                return tool_data.get("name", "")
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for tool names in text
        for tool in self.tools_config:
            if tool["name"].lower() in response.lower():
                return tool["name"]
        
        return ""
    
    def _check_format_correctness(self, response: str) -> bool:
        """Check if response has correct tool call format."""
        # Check for proper tool call tags
        if "[TOOL_CALL]" not in response or "[/TOOL_CALL]" not in response:
            return False
        
        # Extract tool call content
        tool_call_pattern = r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if not matches:
            return False
        
        try:
            # Try to parse as JSON
            tool_data = json.loads(matches[0])
            
            # Check required fields
            if "name" not in tool_data or "parameters" not in tool_data:
                return False
            
            # Check if tool name is valid
            tool_names = [tool["name"] for tool in self.tools_config]
            if tool_data["name"] not in tool_names:
                return False
            
            return True
            
        except json.JSONDecodeError:
            return False
    
    def _check_execution_success(self, response: str) -> bool:
        """Check if tool execution would be successful."""
        if not self._check_format_correctness(response):
            return False
        
        # Extract and execute tool call
        tool_call_pattern = r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]'
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if matches:
            try:
                tool_data = json.loads(matches[0])
                tool_name = tool_data["name"]
                parameters = tool_data["parameters"]
                
                # Execute tool
                result = self.tool_executor.execute_tool(tool_name, parameters)
                
                # Check if execution was successful (no error)
                return "error" not in result
                
            except Exception:
                return False
        
        return False
    
    def evaluate_specific_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate on specific test cases."""
        results = []
        
        for case in test_cases:
            prompt = case["prompt"]
            expected_tool = case["expected_tool"]
            expected_params = case.get("expected_params", {})
            
            response = self._generate_response(prompt)
            predicted_tool = self._extract_tool_name(response)
            
            result = {
                "prompt": prompt,
                "response": response,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "tool_correct": predicted_tool == expected_tool,
                "format_correct": self._check_format_correctness(response),
                "execution_success": self._check_execution_success(response)
            }
            
            results.append(result)
        
        # Calculate summary statistics
        summary = {
            "total_cases": len(results),
            "tool_accuracy": sum(r["tool_correct"] for r in results) / len(results),
            "format_accuracy": sum(r["format_correct"] for r in results) / len(results),
            "execution_accuracy": sum(r["execution_success"] for r in results) / len(results),
            "details": results
        }
        
        return summary


def create_test_cases() -> List[Dict[str, Any]]:
    """Create standard test cases for evaluation."""
    return [
        {
            "prompt": "Human: What is 25 * 4?",
            "expected_tool": "calculator",
            "expected_params": {"expression": "25 * 4"}
        },
        {
            "prompt": "Human: What's the weather like in Paris?",
            "expected_tool": "weather",
            "expected_params": {"location": "Paris"}
        },
        {
            "prompt": "Human: Search for information about machine learning",
            "expected_tool": "search",
            "expected_params": {"query": "machine learning"}
        },
        {
            "prompt": "Human: Calculate 100 divided by 5",
            "expected_tool": "calculator",
            "expected_params": {"expression": "100 / 5"}
        },
        {
            "prompt": "Human: How's the weather in Tokyo today?",
            "expected_tool": "weather",
            "expected_params": {"location": "Tokyo"}
        }
    ]
