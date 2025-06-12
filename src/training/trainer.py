"""Training module for tool use models."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig, DPOTrainer
import wandb

from ..tools.executor import ToolExecutor


logger = logging.getLogger(__name__)


class ToolTrainer:
    """Main trainer class for tool use models."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        train_dataset: Dataset, 
        eval_dataset: Dataset,
        output_dir: Path
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        # Initialize model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Initialize tool executor for evaluation
        self.tool_executor = ToolExecutor(config["tools"])
        
        # Training method
        self.training_method = config["training"]["method"]
        
        logger.info(f"Initialized trainer with method: {self.training_method}")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer."""
        model_name = self.config["model"]["name"]
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", False)
        )
        
        # Add special tokens for tool calls
        special_tokens = {
            "additional_special_tokens": [
                "[TOOL_CALL]", "[/TOOL_CALL]", 
                "[RESULT]", "[/RESULT]"
            ]
        }
        
        tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load and prepare model."""
        model_config = self.config["model"]
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
            device_map=model_config.get("device_map", "auto")
        )
        
        # Resize embeddings for new tokens
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if specified
        if self.config["training"].get("use_lora", True):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config["training"].get("lora_r", 16),
                lora_alpha=self.config["training"].get("lora_alpha", 32),
                lora_dropout=self.config["training"].get("lora_dropout", 0.1),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model based on the specified method."""
        if self.training_method == "sft":
            self._train_sft(resume_from_checkpoint)
        elif self.training_method == "ppo":
            self._train_ppo(resume_from_checkpoint)
        elif self.training_method == "dpo":
            self._train_dpo(resume_from_checkpoint)
        elif self.training_method == "teacher_mode":
            self._train_teacher_mode(resume_from_checkpoint)
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")
    
    def _train_sft(self, resume_from_checkpoint: Optional[str] = None):
        """Supervised fine-tuning."""
        logger.info("Starting supervised fine-tuning...")
        
        # Tokenize datasets
        tokenized_train = self._tokenize_dataset(self.train_dataset)
        tokenized_eval = self._tokenize_dataset(self.eval_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config["training"].get("num_epochs", 3),
            per_device_train_batch_size=self.config["training"].get("batch_size", 4),
            per_device_eval_batch_size=self.config["training"].get("eval_batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            learning_rate=self.config["training"].get("learning_rate", 5e-5),
            warmup_steps=self.config["training"].get("warmup_steps", 100),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config.get("wandb", {}).get("enabled") else None,
            dataloader_pin_memory=False,
            fp16=True,
            max_grad_norm=1.0
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
    
    def _train_ppo(self, resume_from_checkpoint: Optional[str] = None):
        """PPO training with tool success rewards."""
        logger.info("Starting PPO training...")
        
        # PPO Config
        ppo_config = PPOConfig(
            model_name=self.config["model"]["name"],
            learning_rate=self.config["training"].get("learning_rate", 1e-5),
            batch_size=self.config["training"].get("batch_size", 4),
            mini_batch_size=self.config["training"].get("mini_batch_size", 2),
            ppo_epochs=self.config["training"].get("ppo_epochs", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            seed=self.config.get("seed", 42)
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=None,  # Will create reference model automatically
            tokenizer=self.tokenizer,
            dataset=self.train_dataset
        )
        
        # Training loop
        for epoch in range(self.config["training"].get("num_epochs", 1)):
            for batch in ppo_trainer.dataloader:
                # Generate responses
                query_tensors = batch["input_ids"]
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    length_sampler=lambda: self.config["training"].get("max_length", 512),
                    batch_size=len(query_tensors)
                )
                
                # Calculate rewards based on tool execution success
                rewards = []
                for response in response_tensors:
                    decoded_response = self.tokenizer.decode(response, skip_special_tokens=False)
                    reward = self._calculate_tool_reward(decoded_response)
                    rewards.append(torch.tensor(reward))
                
                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log stats
                if wandb.run is not None:
                    wandb.log(stats)
        
        # Save model
        ppo_trainer.save_model(str(self.output_dir))
    
    def _train_dpo(self, resume_from_checkpoint: Optional[str] = None):
        """Direct Preference Optimization training."""
        logger.info("Starting DPO training...")
        
        # For DPO, we need preference pairs
        # This is a simplified implementation
        preference_dataset = self._create_preference_dataset()
        
        # DPO Trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config["training"].get("num_epochs", 1),
                per_device_train_batch_size=self.config["training"].get("batch_size", 4),
                learning_rate=self.config["training"].get("learning_rate", 5e-7),
                logging_steps=10,
                save_steps=500,
                report_to="wandb" if self.config.get("wandb", {}).get("enabled") else None
            ),
            beta=self.config["training"].get("dpo_beta", 0.1),
            train_dataset=preference_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train
        dpo_trainer.train()
        
        # Save model
        dpo_trainer.save_model()
    
    def _train_teacher_mode(self, resume_from_checkpoint: Optional[str] = None):
        """Teacher mode training (Toolformer-style)."""
        logger.info("Starting teacher mode training...")
        
        # This combines SFT with self-supervised learning
        # The data generation already handles teacher mode data creation
        self._train_sft(resume_from_checkpoint)
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize a dataset."""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["training"].get("max_length", 512),
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _calculate_tool_reward(self, response: str) -> float:
        """Calculate reward based on tool execution success."""
        try:
            # Extract tool calls from response
            if "[TOOL_CALL]" in response and "[/TOOL_CALL]" in response:
                # Successfully formatted tool call
                reward = 1.0
                
                # Additional reward for successful execution
                # (This would need actual tool execution logic)
                # For now, just check format
                if "parameters" in response and "name" in response:
                    reward += 0.5
            else:
                # No tool call or malformed
                reward = -0.5
            
            return reward
            
        except Exception:
            return -1.0
    
    def _create_preference_dataset(self) -> Dataset:
        """Create preference dataset for DPO."""
        # This is a simplified implementation
        # In practice, you'd want human preferences or model-based ranking
        
        preference_data = []
        
        for example in self.train_dataset.select(range(min(100, len(self.train_dataset)))):
            # Create a "good" and "bad" version
            good_response = example["text"]
            
            # Create a bad version by removing tool formatting
            bad_response = good_response.replace("[TOOL_CALL]", "").replace("[/TOOL_CALL]", "")
            
            preference_data.append({
                "prompt": example["text"].split("Assistant:")[0] if "Assistant:" in example["text"] else "",
                "chosen": good_response,
                "rejected": bad_response
            })
        
        return Dataset.from_list(preference_data)
