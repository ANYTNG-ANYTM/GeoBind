#!/usr/bin/env python3
"""
GeoBind API Service Entrypoint

Start the FastAPI REST API service for binding predictions.
Usage:
    python run_api.py              # Default (port 8000)
    python run_api.py --port 9000  # Custom port
"""

import sys
import argparse
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.api.phase4_api_service import app
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="GeoBind API Service")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()
    
    print(f"🚀 Starting GeoBind API on {args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "src.api.phase4_api_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
