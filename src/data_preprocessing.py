import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import logging
import joblib
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the input data structure"""
    required_columns = {
        'Transaction Date and Time',
        'Transaction Amount',
        'Fraud Flag or Label',
        'Transaction Source',
        'Card Type'
    }
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    return True, ""

def preprocess_data(input_path: str, output_path: str) -> bool:
    try:
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)

        is_valid, validation_msg = validate_data(data)
        if not is_valid:
            logger.error(f"Data validation failed: {validation_msg}")
            return False

        # Drop rows with nulls in required fields
        critical_columns = [
            'Transaction Date and Time',
            'Transaction Amount',
            'Fraud Flag or Label',
            'Transaction Source',
            'Card Type'
        ]
        data.dropna(subset=critical_columns, inplace=True)

        logger.info("Encoding categorical features dynamically...")

        # Encode 'Transaction Source'
        ts_encoder = LabelEncoder()
        data['Transaction Source'] = ts_encoder.fit_transform(data['Transaction Source'])

        # Encode 'Card Type'
        ct_encoder = LabelEncoder()
        data['Card Type'] = ct_encoder.fit_transform(data['Card Type'])

        # Save encoders
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(ts_encoder, "models/transaction_source_encoder.pkl")
        joblib.dump(ct_encoder, "models/card_type_encoder.pkl")
        logger.info("Encoders saved to models/")

        # Normalize 'Transaction Amount'
        logger.info("Scaling 'Transaction Amount'...")
        scaler = StandardScaler()
        data['Transaction Amount'] = scaler.fit_transform(data[['Transaction Amount']])

        # Drop 'Transaction Date and Time'
        data.drop('Transaction Date and Time', axis=1, inplace=True)

        # Rename target
        data.rename(columns={'Fraud Flag or Label': 'Class'}, inplace=True)

        # Save processed data
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        # Save scaler
        joblib.dump(scaler, 'models/amount_scaler.pkl')
        logger.info("Scaler saved to models/amount_scaler.pkl")

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
