from pathlib import Path
import torch

# Model configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # HuggingFace model
DEVICE = DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 512
TEMPERATURE = 0.7   #0 = deterministic, 1 = very creative.
TOP_P = 0.6       # 0.1 more determinastic, 0.9 = more creative.


# paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True, parents=True)
