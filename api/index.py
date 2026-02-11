"""
Vercel Serverless Function Entry Point

This file serves as the entry point for Vercel's Python runtime.
It imports and exposes the FastAPI app from the backend module.

Vercel expects:
- An 'app' variable that is a FastAPI/ASGI application
- This file to be at api/index.py (or similar)
"""

import sys
import os

# Add project root to Python path so imports work correctly
# This is necessary because Vercel runs from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Import the FastAPI app from backend
from backend.main import app

# Vercel looks for an 'app' variable - this is it!
# The import above already creates it, but we can be explicit:
app = app
