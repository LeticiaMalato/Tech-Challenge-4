"""Split the original dataset into reference and production (raw) datasets."""

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitoring import config
from credit_monitoring.observability.logging import pipeline_step


def main() -> None:
    with pipeline_step("split") as step:
        _split(step)


def _split(step: dict) -> None:
    # 1. Read the original dataset (path from config)
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # 2. Stratified split: PRODUCTION_SIZE for production, RANDOM_STATE for reproducibility
    reference_df, production_df = train_test_split(
        df,
        test_size=config.PRODUCTION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_COLUMN],
    )

    # 3. Save both datasets (no index column)
    reference_df.to_csv(config.REFERENCE_RAW_PATH, index=False)
    production_df.to_csv(config.PRODUCTION_RAW_PATH, index=False)

    # 4. Evidence: shapes, TARGET proportion and shared IDs (expected: 0)
    print(f"Reference dataset shape: {reference_df.shape}")
    print(f"Production dataset shape: {production_df.shape}")
    print(
        f"Reference TARGET proportion: {reference_df[config.TARGET_COLUMN].mean():.4%}"
    )
    print(
        f"Production TARGET proportion: {production_df[config.TARGET_COLUMN].mean():.4%}"
    )
    shared_ids = set(reference_df[config.ID_COLUMN]) & set(
        production_df[config.ID_COLUMN]
    )
    print(f"Shared IDs: {len(shared_ids)}")
    step["rows"] = int(len(df))
    step["reference_rows"] = int(len(reference_df))
    step["production_rows"] = int(len(production_df))


if __name__ == "__main__":
    main()
