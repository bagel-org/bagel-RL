"""Data generation for tool use training."""

import json
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from datasets import Dataset, load_dataset
import pandas as pd
from ..tools.executor import ToolExecutor
import os
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates training data for tool use from various sources."""
    
    def __init__(self, data_config: Dict[str, Any], tools_config: List[Dict[str, Any]], tokenizer_config: List[Dict[str, Any]]):
        self.data_config = data_config
        self.tools_config = tools_config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["name"], trust_remote_code=tokenizer_config['trust_remote_code'])
        
       
        self.tool_executor = ToolExecutor(tools_config)
        self.strategy = data_config["strategy"]
        self.generation_type = data_config["generation_type"]

        
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and evaluation datasets."""
        if self.strategy == "toolbench" and self.generation_type.lower()=='real':
            return self._prepare_real_toolbench_data()
        elif self.strategy == "toolbench" and self.generation_type.lower()=='synthetic':
            return self._prepare_synthetic_toolbench_data()
        elif self.strategy == "teacher_mode":
            return self._prepare_teacher_mode_data()
        elif self.strategy == "manual_templates":
            return self._prepare_manual_template_data()
        else:
            raise ValueError(f"Unknown data strategy: {self.strategy}")
    

    def _download_from_google_drive(self, folder_url, destination_dir):

        import gdown, pathlib, zipfile

        destination_dir = pathlib.Path(destination_dir)

        files = gdown.download_folder(
            url = folder_url,
            quiet = False,
            use_cookies = False,
            output = destination_dir.as_posix()

        )

        zip_path = next(p for p in files if p.endswith('data.zip'))

        print("✔ downloaded", zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destination_dir.as_posix()+"/data")
        return 



        
        
    def _prepare_synthetic_toolbench_data(self)->Tuple[Dataset, Dataset]:

        "Get the synthetic tool bench data"

        logger.info("Generating synthetic tool bench data...")

        synthetic_data = self._generate_synthetic_toolbench_data()

        return self._split_dataset(synthetic_data)





    
    def _prepare_real_toolbench_data(self) -> Tuple[Dataset, Dataset]:
        """Get toolbench data."""
        logger.info("Obtaining toolbench data...")

        assistant_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        # synthetic_data = self._generate_synthetic_toolbench_data()

        #download the data from google drive link 
        destination_dir = './data/toolbench/'
        if not os.path.exists(destination_dir):
            folder_url = 'https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL'
            destination_dir = './data/toolbench/'
            self._download_from_google_drive(folder_url, destination_dir)
        
        #loading the toolbench data
        data = load_dataset("json", data_files="./data/toolbench/data/data/toolllama_G123_dfs_train.json")["train"]
        
        data = data.shuffle(seed=42).select(range(self.data_config["max_samples"]))
     
        def to_messages(conv):
             # Map any role names that can appear in ToolBench/Qwen
            role_map = {
                "system": "system",
                "user": "user",
                "assistant": "assistant",
                "tool": "tool",           # tool_response in some repos
                "function": "tool",       # treat function output the same as tool
                "tool_response": "tool",  # safety net for other dumps
                "tool_call": "assistant", # if your dump keeps the call separate
            }

            unknown = {m["from"] for m in conv} - role_map.keys()
            if unknown:                       # fail fast if you meet something new
                raise ValueError(f"Unknown role(s): {unknown}")

            return [
                {"role": role_map[m["from"]], "content": m["value"]}
                for m in conv
            ]
        
        def tokenize(sample):
            msgs = to_messages(sample["conversations"])
            chat_text = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=False)  # Qwen-3 Jinja template:contentReference[oaicite:1]{index=1}
            
            
            ids = self.tokenizer(chat_text, return_tensors="pt").input_ids[0]
            labels = ids.clone()

            # *** non-assistant masking ***
            ptr = 0
            for msg in msgs:
                n = len(self.tokenizer(msg["content"]).input_ids) + 1  # +EOS
                if msg["role"] != "assistant":
                    labels[ptr:ptr+n] = -100        # ignore in loss
                ptr += n
            sample["input_ids"], sample["labels"] = ids, labels
            return sample

        tokenised = data.map(tokenize, remove_columns=data.column_names)
        tokenised = tokenised.shuffle(seed=42).train_test_split(test_size=0.01)

        dataset_train = tokenised["train"]
        dataset_eval = tokenised['test']

        return dataset_train, dataset_eval


    
    
    def _prepare_teacher_mode_data(self) -> Tuple[Dataset, Dataset]:
        """Generate data using teacher mode (Toolformer-style)."""
        logger.info("Generating teacher mode data...")
        
        data = []
        for _ in range(self.data_config.get("max_samples", 100)):
            conversation = self._generate_teacher_mode_example()
            data.append(conversation)
        
        logger.info(f"Generated {len(data)} teacher mode examples")
        return self._split_dataset(data)
    
    def _prepare_manual_template_data(self) -> Tuple[Dataset, Dataset]:
        """Generate data from manual templates with paraphrasing."""
        logger.info("Generating data from manual templates...")
        
        canonical_examples = self._create_canonical_examples()
        
        bootstrapped_data = []
        for example in canonical_examples:
            bootstrapped_data.append(example)
            paraphrases = self._simple_paraphrase(example)
            bootstrapped_data.extend(paraphrases)
        
        logger.info(f"Generated {len(bootstrapped_data)} template-based examples")
        return self._split_dataset(bootstrapped_data)
    
    def _generate_synthetic_toolbench_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic ToolBench-style data."""
        data = []
        
        for tool in self.tools_config:
            for i in range(20):
                conversation = self._create_tool_conversation(tool)
                data.append({"text": conversation, "tool_name": tool["name"]})
        
        return data
    
    def _create_tool_conversation(self, tool: Dict[str, Any]) -> str:
        """Create a conversation that uses a specific tool."""
        tool_name = tool["name"]
        
        user_queries = {
            "calculator": ["What's 15 * 24?", "Can you calculate 45 + 67 - 12?"],
            "weather": ["What's the weather like in New York?", "Check London weather"],
            "search": ["Search for Python tutorials", "Find ML information"]
        }
        
        queries = user_queries.get(tool_name, [f"Use {tool_name}"])
        user_query = random.choice(queries)
        
        if tool_name == "calculator":
            expression = random.choice(["15 * 24", "45 + 67 - 12", "(100 / 5) * 3"])
            params = {"expression": expression}
        elif tool_name == "weather":
            location = random.choice(["New York", "London", "Tokyo"])
            params = {"location": location}
        elif tool_name == "search":
            query = random.choice(["Python tutorials", "machine learning"])
            params = {"query": query}
        else:
            params = {}
        
        result = self.tool_executor.execute_tool(tool_name, params)
        tool_call = json.dumps({"name": tool_name, "parameters": params})
        result_str = json.dumps(result)
        
        conversation = f"""Human: {user_query}
Assistant: {tool_call}
"""
        
        return conversation
    
    def _format_toolbench_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format ToolBench data to our internal format."""
        formatted_data = []
        
        for example in data:
            conversation = example["conversation"]
            tool_name = example["tool_name"]
            
            # Simple format: just pass through
            formatted_data.append({
                "input": conversation,
                "output": tool_name
            })
        
        return formatted_data
    
    def _split_dataset(self, data: List[Dict[str, Any]]) -> Tuple[Dataset, Dataset]:
        """Split data into training and evaluation sets."""
        df = pd.DataFrame(data)
        
        # 80-20 train-test split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, test_dataset
    
    def _generate_base_conversations(self) -> List[Dict[str, Any]]:
        """Generate base conversations for teacher mode."""
        base_conversations = []
        
        for tool in self.tools_config:
            tool_name = tool["name"]
            description = tool["description"]
            
            # Simple static prompts for now
            conversation = f"Use the {tool_name} to {description.lower()}"
            base_conversations.append({
                "conversation": conversation,
                "tool_name": tool_name
            })
        
        return base_conversations
    
    def _insert_tool_calls(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert tool calls into a base conversation (teacher model style)."""
        tool_name = conversation["tool_name"]
        base_convo = conversation["conversation"]
        
        # Naive insertion of tool call - just for demonstration
        if tool_name in base_convo:
            return conversation  # Nothing to change
        
        # Insert the tool call before the user query
        user_query = f"Please {base_convo}"
        tool_call = json.dumps({"name": tool_name, "parameters": {}})
        full_conversation = f"""Human: {user_query}
Assistant: {tool_call}
"""
        
        return {
            "conversation": full_conversation,
            "tool_name": tool_name
        }
    
    def _generate_teacher_mode_example(self) -> Dict[str, Any]:
        """Generate a teacher mode example with tool insertion."""
        # Start with a base conversation
        topics = [
            "I need help with calculations",
            "Can you tell me about the weather?",
            "I want to search for information"
        ]
        
        topic = random.choice(topics)
        tool = random.choice(self.tools_config)
        
        # Generate conversation with tool insertion
        conversation = self._create_tool_conversation(tool)
        
        return {"text": conversation, "tool_name": tool["name"]}
    
    def _create_canonical_examples(self) -> List[Dict[str, Any]]:
        """Create canonical examples for each tool."""
        examples = []
        
        templates = {
            "calculator": [
                "Calculate {expression}",
                "What is {expression}?",
                "Compute {expression}",
                "Solve {expression}",
                "Find the result of {expression}"
            ],
            "weather": [
                "What's the weather in {location}?",
                "Check weather for {location}",
                "Weather forecast for {location}",
                "How's the weather in {location}?",
                "Tell me about {location} weather"
            ],
            "search": [
                "Search for {query}",
                "Find information about {query}",
                "Look up {query}",
                "Research {query}",
                "Get results for {query}"
            ]
        }
        
        for tool in self.tools_config:
            tool_name = tool["name"]
            tool_templates = templates.get(tool_name, [f"Use {tool_name}"])
            
            # Create 10 canonical examples per tool
            for i in range(10):
                template = random.choice(tool_templates)
                
                if tool_name == "calculator":
                    expressions = ["2 + 3", "10 * 5", "100 / 4", "15 - 7", "2 ** 3"]
                    expression = random.choice(expressions)
                    user_query = template.format(expression=expression)
                    params = {"expression": expression}
                elif tool_name == "weather":
                    locations = ["Paris", "Tokyo", "Sydney", "Berlin", "Cairo"]
                    location = random.choice(locations)
                    user_query = template.format(location=location)
                    params = {"location": location}
                elif tool_name == "search":
                    queries = ["Python", "AI", "cooking", "travel", "science"]
                    query = random.choice(queries)
                    user_query = template.format(query=query)
                    params = {"query": query}
                else:
                    user_query = template
                    params = {}
                
                result = self.tool_executor.execute_tool(tool_name, params)
                tool_call = json.dumps({"name": tool_name, "parameters": params})
                result_str = json.dumps(result)
                
                conversation = f"""Human: {user_query}
Assistant: I'll help you with that. Let me use the {tool_name} function.

[TOOL_CALL]{tool_call}[/TOOL_CALL]

{result_str}

Based on the result, the answer is {result.get('result', 'processed successfully')}."""
                
                examples.append({"text": conversation, "tool_name": tool_name})
        
        return examples
    
    def _simple_paraphrase(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate simple paraphrases of an example."""
        paraphrases = []
        original_text = example["text"]
        
        # Simple paraphrasing by replacing words
        replacements = {
            "Calculate": "Compute",
            "What is": "What's",
            "Can you": "Could you",
            "Find": "Get",
            "Search for": "Look up",
            "weather": "forecast",
            "information": "info"
        }
        
        # Generate 3 paraphrases
        for i in range(3):
            paraphrased = original_text
            
            # Apply random replacements
            for original, replacement in replacements.items():
                if original in paraphrased and random.random() < 0.5:
                    paraphrased = paraphrased.replace(original, replacement)
            
            if paraphrased != original_text:
                paraphrases.append({
                    "text": paraphrased,
                    "tool_name": example["tool_name"]
                })
        
        return paraphrases
    
    def _split_dataset(self, data: List[Dict[str, Any]]) -> Tuple[Dataset, Dataset]:
        """Split data into train and eval datasets."""
        random.shuffle(data)
        
        train_split = self.data_config.get("train_split", 0.8)
        split_idx = int(len(data) * train_split)
        
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        # Ensure we have at least some eval data
        if len(eval_data) == 0 and len(train_data) > 1:
            eval_data = [train_data.pop()]
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        return train_dataset, eval_dataset
    
    def _log_dataset_sample(self, dataset: Dataset, num_samples: int = 3):
        """Log a few samples from the dataset."""
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            logger.info(f"Sample {i+1}: {example}")
