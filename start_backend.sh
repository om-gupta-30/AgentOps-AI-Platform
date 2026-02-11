#!/bin/bash

# AgentOps AI - Backend Startup Script
# This script properly starts the backend with correct paths and environment

cd "$(dirname "$0")"

echo "🚀 Starting AgentOps AI Backend..."
echo ""

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo ""
    echo "Please install dependencies first:"
    echo "  pip3 install -r requirements.txt"
    echo ""
    exit 1
fi

# Load environment variables
if [ -f ".env.local" ]; then
    echo "✅ Loading environment variables from .env.local"
    set -a
    source .env.local
    set +a
else
    echo "⚠️  Warning: .env.local not found"
    echo "   Create it from .env.example"
    echo ""
    echo "Quick fix:"
    echo "  cp .env.example .env.local"
    echo "  nano .env.local  # Add your GOOGLE_API_KEY"
    echo ""
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set in .env.local"
    echo ""
    echo "Edit .env.local and add your key:"
    echo '  GOOGLE_API_KEY="your_key_here"'
    echo ""
    exit 1
fi

# Set Python path to include src directory
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"

echo "✅ Python path configured"
echo "✅ Environment variables loaded"
echo ""
echo "📡 Starting FastAPI server on http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo "   Health check: http://localhost:8000/health"
echo "   Press Ctrl+C to stop"
echo ""

# Start uvicorn from the project root
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
