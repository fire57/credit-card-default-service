from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"

TARGET_COLUMN = "default.payment.next.month"
ID_COLUMN = "ID"

FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

MODEL_FILENAMES = {
    "v1": "model_v1.joblib",
    "v2": "model_v2.joblib",
}
