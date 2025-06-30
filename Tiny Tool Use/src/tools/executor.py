"""Tool management and execution utilities."""

import logging
import subprocess
from typing import Dict, Any, List, Optional
import requests


logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles execution of external tools."""
    
    def __init__(self, tools_config: List[Dict[str, Any]]):
        self.tools = {tool["name"]: tool for tool in tools_config}
        logger.info(f"Initialized {len(self.tools)} tools: {list(self.tools.keys())}")
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        try:
            # Handle different tool types
            if tool.get("type") == "api":
                return self._execute_api_tool(tool, parameters)
            elif tool.get("type") == "function":
                return self._execute_function_tool(tool, parameters)
            elif tool.get("type") == "command":
                return self._execute_command_tool(tool, parameters)
            else:
                # Default to function execution
                return self._execute_function_tool(tool, parameters)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": str(e)}
    
    def _execute_api_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API-based tool."""
        url = tool["url"]
        method = tool.get("method", "POST")
        headers = tool.get("headers", {})
        
        if method.upper() == "POST":
            response = requests.post(url, json=parameters, headers=headers)
        elif method.upper() == "GET":
            response = requests.get(url, params=parameters, headers=headers)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        
        response.raise_for_status()
        return response.json()
    
    def _execute_function_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a built-in function tool."""
        function_name = tool.get("function", tool["name"])
        
        # Built-in calculator example
        if function_name == "calculator":
            return self._calculator(parameters)
        elif function_name == "weather":
            return self._weather_mock(parameters)
        elif function_name == "search":
            return self._search_mock(parameters)
        else:
            return {"error": f"Function '{function_name}' not implemented"}
    
    def _execute_command_tool(self, tool: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command-line tool."""
        command = tool["command"].format(**parameters)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
    
    def _calculator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple calculator implementation."""
        expression = parameters.get("expression", "")
        
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    def _weather_mock(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock weather API."""
        location = parameters.get("location", "Unknown")
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
    
    def _search_mock(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock search API."""
        query = parameters.get("query", "")
        return {
            "query": query,
            "results": [
                {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                {"title": f"Result 2 for {query}", "url": "https://example.com/2"}
            ]
        }
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific tool."""
        if tool_name in self.tools:
            return self.tools[tool_name]
        return None
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())
