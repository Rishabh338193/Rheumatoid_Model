#!/bin/bash

# Rheumatoid Arthritis Prediction - Simple Run Script
# Run this to start the entire project with one command

echo "=========================================="
echo "🏥 RA Prediction System - Starting"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies silently
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt 2>/dev/null

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Start Backend API in background
echo -e "${YELLOW}Starting Backend API...${NC}"
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

sleep 2

# Start Streamlit Frontend in background
echo -e "${YELLOW}Starting Frontend (Streamlit)...${NC}"
cd frontend
python3 -m streamlit run streamlit_app.py --logger.level=error &
FRONTEND_PID=$!
cd ..

sleep 3

# Start static HTML server (frontend) on port 8080
echo -e "${YELLOW}Starting static HTML server (frontend) on port 8080...${NC}"
cd frontend
python3 -m http.server 8080 &
STATIC_PID=$!
cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Project is Running!${NC}"
echo "=========================================="
echo ""
echo "📱 Open your browser:"
echo -e "   ${YELLOW}Frontend:  http://localhost:8501${NC}"
echo -e "   ${YELLOW}API Docs:  http://127.0.0.1:5001/docs${NC}"
echo ""
echo "To stop the project:"
echo -e "   ${YELLOW}Press Ctrl+C here${NC}"
echo ""
echo "=========================================="
echo ""

# Wait for Ctrl+C to stop both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo 'Project stopped.'; exit 0" SIGINT

# Keep script running
wait
