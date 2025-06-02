import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import joblib
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the input data structure"""
    required_columns = {'Time', 'Amount', 'Class'} | {f'V{i}' for i in range(1, 29)}
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    if data.isnull().any().any():
        return False, "Data contains null values"
    return True, ""

def preprocess_data(input_path: str, output_path: str) -> bool:
    """
    Preprocess credit card transaction data
    
    Args:
        input_path: Path to raw CSV data
        output_path: Path to save processed data
        
    Returns:
        bool: True if preprocessing succeeded
    """
    try:
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        
        # Data validation
        is_valid, validation_msg = validate_data(data)
        if not is_valid:
            logger.error(f"Data validation failed: {validation_msg}")
            return False
        
        # Preprocessing pipeline
        logger.info("Preprocessing data...")

        # 1. Normalize Amount
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data[['Amount']])

        # 2. Drop Time column
        data.drop('Time', axis=1, inplace=True)

        # 3. Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. Save processed data
        data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        # 5. Save scaler for inference
        joblib.dump(scaler, 'models/amount_scaler.pkl')
        logger.info("Amount scaler saved to models/amount_scaler.pkl")

        return True

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = preprocess_data(
        input_path="data/raw/creditcard.csv",
        output_path="data/processed/processed.csv"
    )
    
    if not success:
        logger.error("Data preprocessing pipeline failed")
        exit(1)
