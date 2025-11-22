import sys
from pathlib import Path

# Add repo root to sys.path for test imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
