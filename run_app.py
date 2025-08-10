"""Streamlit app launcher with proper path setup."""
import os
import sys

# Ensure we're in the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # This script is already in the project root
os.chdir(project_root)

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path includes project root: {project_root in sys.path}")

# Now import and run the main app
if __name__ == "__main__":
    # Import streamlit and run the app
    import subprocess
    import sys
    
    # Run streamlit with the correct app file
    app_file = os.path.join("app", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])
