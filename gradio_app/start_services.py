#!/usr/bin/env python3
"""
Startup script for Faster Whisper services
"""
import argparse
import subprocess
import sys
import os

def start_gradio():
    """Start the Gradio UI"""
    print("Starting Gradio UI on http://localhost:7860")
    subprocess.run([sys.executable, "app.py"], cwd=os.path.dirname(__file__))

def start_api():
    """Start the Flask API server"""
    print("Starting API server on http://localhost:8000")
    subprocess.run([sys.executable, "api_server.py"], cwd=os.path.dirname(__file__))

def start_both():
    """Start both services (requires separate terminals)"""
    print("To start both services, run these commands in separate terminals:")
    print("Terminal 1: python start_services.py --gradio")
    print("Terminal 2: python start_services.py --api")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Faster Whisper services")
    parser.add_argument("--gradio", action="store_true", help="Start Gradio UI")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--both", action="store_true", help="Show commands to start both")
    
    args = parser.parse_args()
    
    if args.gradio:
        start_gradio()
    elif args.api:
        start_api()
    elif args.both:
        start_both()
    else:
        print("Choose a service to start:")
        print("  --gradio : Start Gradio UI")
        print("  --api    : Start API server")
        print("  --both   : Show commands to start both")
