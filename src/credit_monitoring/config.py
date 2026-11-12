"""Project configuration: paths and parameters.""" 
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Data files
TRAIN_DATA_PATH = DATA_DIR / "application_train.csv"
REFERENCE_RAW_PATH = DATA_DIR / "reference_raw.csv"
PRODUCTION_RAW_PATH = DATA_DIR / "production_raw.csv"

# Split parameters
PRODUCTION_SIZE = 0.2
RANDOM_STATE = 42

# Dataset columns
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Contract limits
DAYS_PER_YEAR = 365.25
MIN_AGE = 18
MAX_AGE = 115

# Data files (after the contract)
REFERENCE_CLEAN_PATH = DATA_DIR / "reference_clean.csv"

# Contract parameters
DAYS_EMPLOYED_SENTINEL = 365243   # "not employed" code (pensioners / unemployed)
CAR_AGE_ALERT = 60                # just below the suspicious peaks at 64-65 years
INCOME_ALERT_THRESHOLD = 900_000      # fill in with the value from step 2