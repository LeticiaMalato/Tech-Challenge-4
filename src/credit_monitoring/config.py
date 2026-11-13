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
DAYS_EMPLOYED_SENTINEL = 365243  # "not employed" code (pensioners / unemployed)
CAR_AGE_ALERT = 60  # just below the suspicious peaks at 64-65 years
INCOME_ALERT_THRESHOLD = 900_000  # reference p99.9, computed in eda_reference.ipynb
# Feature bins (fixed, so drift stays visible in production)
CAR_AGE_BINS = [0, 5, 10, 15, 20, float("inf")]
CAR_AGE_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]
EMPLOYMENT_BINS = [0, 1, 3, 5, 10, float("inf")]
EMPLOYMENT_LABELS = ["0-1", "1-3", "3-5", "5-10", "10+"]
HOUSING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")

# Decision policy (business assumption, not learned from data)
APPROVAL_RATE = 0.90  # approve the 90% lowest-risk clients
APPROVAL_RATES_TO_COMPARE = [0.80, 0.85, 0.90, 0.95, 0.99]

# Evidence (versioned: aggregated numbers only, no personal data)
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"

# Model
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_logreg.joblib"
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AGE_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "REGION_RATING_CLIENT_W_CITY",
    "HOUSING_INFO_MISSING",
]
CATEGORICAL_FEATURES = [
    "EMPLOYMENT_GROUP",
    "CAR_AGE_GROUP",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

# Drift simulation (explicit, reproducible scenario)
PRODUCTION_DRIFTED_PATH = DATA_DIR / "production_drifted.csv"
SIMULATION_SUMMARY_PATH = REPORTS_DIR / "simulation_summary.json"
INFLATION_MIN = 0.0  # per-client income increase between 0% and 40%...
INFLATION_MAX = 0.4  # ...so +20% on average
UNEMPLOYMENT_SHARE = 0.15  # share of "Working" clients who lose their job
UNEMPLOYED_DEFAULT_RATE = 0.294  # reference "Unemployed": 5 of 17 (95% CI ~13%-53%)

# Production scoring
PREDICTIONS_CONTROL_PATH = (
    DATA_DIR / "predictions_control.csv"
)  # row-level: not versioned
PREDICTIONS_DRIFTED_PATH = (
    DATA_DIR / "predictions_drifted.csv"
)  # row-level: not versioned
PRODUCTION_SUMMARY_PATH = (
    REPORTS_DIR / "production_summary.json"
)  # aggregated: versioned
PRICE_INFLATION = 0.20  # prices rise 20% for everyone (inflation index)
PRICE_COLUMNS = ["AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"]

# Drift detection
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"
PSI_MODERATE = 0.10  # credit-scoring convention: below this, the variable is stable
PSI_SIGNIFICANT = 0.25  # above this, the shift is significant
PSI_BINS = 10  # numeric variables: deciles of the reference

# Concept drift scenario: same features, higher risk for clients with high payment burden
PRODUCTION_CONCEPT_PATH = DATA_DIR / "production_concept.csv"
PREDICTIONS_CONCEPT_PATH = DATA_DIR / "predictions_concept.csv"
COMMITMENT_QUANTILE = 0.80  # top 20% annuity/income ratio in the REFERENCE
CONCEPT_DEFAULT_MULTIPLIER = 2.0  # default rate in that group doubles

# Known categories (observed in the reference EDA); unknown values trigger an alert
KNOWN_CATEGORIES = {
    "NAME_INCOME_TYPE": [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
}
REGION_RATINGS = [1, 2, 3]
