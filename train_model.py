import sys
import os

# Ensure project root on sys.path so `app` can be imported when run directly
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.trained_model import train


if __name__ == "__main__":
    train()


