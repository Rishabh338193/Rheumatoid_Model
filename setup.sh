#!/bin/bash

# Rheumatoid Arthritis Prediction - Setup Script
# This script sets up and runs the complete project

echo "=========================================="
echo "RA Prediction System - Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python installation
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION found${NC}"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Generate dataset
echo "Generating synthetic RA dataset..."
cd data
python3 generate_dataset.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset generated successfully${NC}"
else
    echo -e "${RED}✗ Failed to generate dataset${NC}"
    exit 1
fi
cd ..
echo ""

# Train model
echo "Training Random Forest model..."
cd notebooks
python3 train_model.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model trained successfully${NC}"
else
    echo -e "${RED}✗ Failed to train model${NC}"
    exit 1
fi
cd ..
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo -e "${YELLOW}1. Start Backend API:${NC}"
echo "   cd backend && python3 app.py"
echo ""
echo -e "${YELLOW}2. Open Frontend (choose one):${NC}"
echo "   Option A: Open frontend/index.html in browser"
echo "   Option B: cd frontend && streamlit run streamlit_app.py"
echo ""
echo "=========================================="
