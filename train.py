#!/usr/bin/env python3
"""
LLM Tool Use Training Playground
Main training script supporting multiple training recipes and data sources.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from rich.console import Console

from src.data.data_generator import DataGenerator
from src.training.trainer import ToolTrainer
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train LLM for tool use")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Training method: {config['training']['method']}")
    logger.info(f"Data strategy: {config['data']['strategy']}")
    
    # TensorBoard logging is handled by the trainer itself
    # No additional initialization needed here
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 42))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    try:
        # Initialize data generator
        console.print("🔄 [bold blue]Initializing data generator...[/bold blue]")
        data_generator = DataGenerator(config["data"], config["tools"])
        
        # Generate or load training data
        console.print("📊 [bold blue]Preparing training data...[/bold blue]")
        train_dataset, eval_dataset = data_generator.prepare_datasets()
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Initialize trainer
        console.print("🚀 [bold blue]Initializing trainer...[/bold blue]")
        trainer = ToolTrainer(
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir
        )
        
        # Start training
        console.print("🎯 [bold green]Starting training...[/bold green]")
        trainer.train(resume_from_checkpoint=args.resume)
        
        console.print("✅ [bold green]Training completed successfully![/bold green]")
        
        # Clean up resources
        trainer.cleanup()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        console.print(f"❌ [bold red]Training failed: {str(e)}[/bold red]")
        # Clean up even on failure
        if 'trainer' in locals():
            trainer.cleanup()
        raise
    
    # No cleanup needed for TensorBoard
    # The trainer handles the SummaryWriter lifecycle


if __name__ == "__main__":
    main()
