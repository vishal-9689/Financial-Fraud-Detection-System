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
import os
from pathlib import Path
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Create directories if they don't exist
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and preprocess data with error handling"""
    try:
        data = pd.read_csv("data/raw/creditcard.csv")
        
        # Validate required columns
        if 'Amount' not in data.columns or 'Class' not in data.columns:
            raise ValueError("Missing required columns in dataset")
            
        # Preprocessing
        data['Amount'] = StandardScaler().fit_transform(data[['Amount']])
        data.drop('Time', axis=1, inplace=True)
        
        # Save processed data
        data.to_csv("data/processed/processed.csv", index=False)
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

def train_models(X_train, y_train):
    """Train and save all models with validation"""
    try:
        # 1. Random Forest
        print("Training Random Forest with preprocessing pipeline...")
        pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
        ))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, "models/random_forest_pipeline.pkl")
        
        # 2. XGBoost
        print("\nTraining XGBoost...")
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        xgb.save_model("models/xgboost.model")
        
        # 3. Autoencoder
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
        
        print("\nAll models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    # Load and prepare data
    data = load_data()
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    success = train_models(X_train, y_train)
    
    if success:
        # Basic validation
        rf = joblib.load("models/random_forest.pkl")
        print("\nRandom Forest test accuracy:", rf.score(X_test, y_test))
