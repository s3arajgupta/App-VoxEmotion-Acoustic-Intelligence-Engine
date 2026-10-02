"""VoxEmotion Application Launcher.

Runs the Streamlit audio analysis studio.
"""

import sys
import subprocess
from pathlib import Path


def launch():
    app_file = Path(__file__).parent / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    print(f"Launching VoxEmotion Studio: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    launch()
