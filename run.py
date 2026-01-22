#!/usr/bin/env python3
"""
Simple Project Launcher
Run the RA Prediction System with one command
"""

import subprocess
import sys
import os
import time
import signal

def run_project():
    print("=" * 50)
    print("🏥 RA Prediction System - Starting")
    print("=" * 50)
    print()
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            timeout=120
        )
        print("✓ Dependencies ready")
    except subprocess.TimeoutExpired:
        print("⚠ Installation took longer, continuing...")
    except Exception as e:
        print(f"⚠ Could not install: {e}, continuing...")
    
    print()
    
    # Start backend
    print("🚀 Starting Backend API...")
    backend_proc = subprocess.Popen(
        [sys.executable, "backend/app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    
    time.sleep(3)
    
    # Start frontend
    print("🎨 Starting Frontend (Streamlit)...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--logger.level=error"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    
    time.sleep(3)
    
    print()
    print("=" * 50)
    print("✅ Project is Running!")
    print("=" * 50)
    print()
    print("📱 Open your browser:")
    print("   Frontend:  http://localhost:8501")
    print("   API:       http://127.0.0.1:5001")
    print()
    print("To stop: Press Ctrl+C")
    print("=" * 50)
    print()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n🛑 Stopping project...")
        import os
        import signal as sig_module
        if hasattr(os, 'setsid'):
            os.killpg(os.getpgid(backend_proc.pid), sig_module.SIGTERM)
            os.killpg(os.getpgid(frontend_proc.pid), sig_module.SIGTERM)
        else:
            backend_proc.terminate()
            frontend_proc.terminate()
        time.sleep(1)
        print("✓ Project stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None or frontend_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    run_project()

