"""
main.py
-------
Application entry point.

Run with:
    python main.py
"""

import sys
import os

# Ensure the project root is on the path so all packages resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from ui.ct_simulation_ui import main

if __name__ == "__main__":
    main()
