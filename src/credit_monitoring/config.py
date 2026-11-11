"""Project configuration: paths and parameters.""" 
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Data files
TRAIN_DATA_PATH = DATA_DIR / "application_train.csv"
REFERENCE_RAW_PATH = DATA_DIR / "reference.csv"
PRODUCTION_RAW_PATH = DATA_DIR / "production.csv"

# Split parameters
PRODUCTION_SIZE = 0.2
RANDOM_STATE = 42