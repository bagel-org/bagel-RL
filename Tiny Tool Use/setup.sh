#!/bin/bash

# LLM Tool Use Training Playground Setup Script

set -e

echo "🚀 Setting up LLM Tool Use Training Playground..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs logs

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run a training example: python train.py --config configs/calculator_config.json"
echo "3. Or run the interactive examples: python examples/run_examples.py"
echo ""
echo "For more information, see README.md"
