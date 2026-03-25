#!/usr/bin/env python3
"""
GeoBind Web UI Entrypoint

Start the Streamlit web interface for binding predictions.
Usage:
    python run_ui.py              # Default (port 8501)
    python run_ui.py --port 9001  # Custom port
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="GeoBind Web UI")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    args = parser.parse_args()
    
    print(f"🌐 Starting GeoBind Web UI on {args.host}:{args.port}")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).parent / "src" / "ui" / "phase5_web_frontend.py"),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
