#!/usr/bin/env python3
"""
Evaluation script for trained tool use models.
"""

import argparse
import json
import logging

from src.utils.evaluation import ToolUseEvaluator, create_test_cases
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained tool use model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Custom test cases JSON file"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Initialize evaluator
    evaluator = ToolUseEvaluator(args.model_path, config["tools"])
    
    # Load test cases
    if args.test_cases:
        with open(args.test_cases, 'r') as f:
            test_cases = json.load(f)
    else:
        test_cases = create_test_cases()
    
    logger.info(f"Evaluating model on {len(test_cases)} test cases...")
    
    # Run evaluation
    results = evaluator.evaluate_specific_cases(test_cases)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Tool Accuracy: {results['tool_accuracy']:.3f}")
    print(f"Format Accuracy: {results['format_accuracy']:.3f}")
    print(f"Execution Accuracy: {results['execution_accuracy']:.3f}")
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
