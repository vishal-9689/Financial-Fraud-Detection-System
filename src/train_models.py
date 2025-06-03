import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib
from pathlib import Path
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Create necessary directories
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and preprocess the dataset with updated column names"""
    try:
        data = pd.read_csv("data/raw/creditcard.csv")

        # Drop unused datetime column if present
        columns_to_drop = [
            'Transaction Date and Time',
            'Cardholder Name',
            'Card Number (Hashed or Encrypted)',
            'Merchant Name',
            'Transaction Currency',
            'Card Expiration Date',
            'CVV Code (Hashed or Encrypted)',
            'Transaction Response Code',
            'Transaction ID',
            'Previous Transactions',
            'IP Address',
            'Device Information',
            'User Account Information',
            'Transaction Notes',
            'Transaction Location (City or ZIP Code)',
        ]
        data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

        # Required columns based on your dataset
        required = [
            'Transaction Amount', 'Transaction Source',
            'Card Type', 'Fraud Flag or Label'
        ]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Drop rows with nulls
        data.dropna(inplace=True)

        # Encode categorical features
        transaction_map = {
            val: i for i, val in enumerate(data['Transaction Source'].dropna().unique())
        }
        card_map = {
            val: i for i, val in enumerate(data['Card Type'].dropna().unique())
        }

        data['Transaction Source'] = data['Transaction Source'].map(transaction_map)
        data['Card Type'] = data['Card Type'].map(card_map)

        if data[['Transaction Source', 'Card Type']].isnull().any().any():
            raise ValueError("Unmapped values found in categorical columns after mapping")

        # Scale 'Transaction Amount'
        scaler = StandardScaler()
        data['Transaction Amount'] = scaler.fit_transform(data[['Transaction Amount']])
        joblib.dump(scaler, "models/amount_scaler.pkl")

        # Rename columns to match training code
        data.rename(columns={
            'Transaction Amount': 'Amount',
            'Transaction Source': 'Transaction_Type',
            'Card Type': 'Card_Type',
            'Fraud Flag or Label': 'Class'
        }, inplace=True)

        # Save processed CSV
        data.to_csv("data/processed/processed.csv", index=False)
        return data

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit(1)


def train_models(X_train, y_train):
    """Train and save RandomForest, XGBoost, and Autoencoder models"""
    try:
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        )
        rf.fit(X_train, y_train)
        joblib.dump(rf, "models/random_forest.pkl")

        print("\nTraining XGBoost...")
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        xgb.save_model("models/xgboost.model")

        print("\nTraining Autoencoder...")
        X_train_normal = X_train[y_train == 0]
        input_dim = X_train.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(14, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)

        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        autoencoder.fit(
            X_train_normal, X_train_normal,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        save_model(autoencoder, "models/autoencoder.h5")

        print("\n✅ All models trained and saved successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    data = load_data()
    X = data.drop(columns=['Class'])
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    success = train_models(X_train, y_train)

    if success:
        try:
            rf = joblib.load("models/random_forest.pkl")
            print("\n📊 Random Forest Test Accuracy:", rf.score(X_test, y_test))
        except:
            print("⚠️ Unable to evaluate Random Forest.")
