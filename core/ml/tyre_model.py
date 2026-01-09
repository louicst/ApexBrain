# core/ml/tyre_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

class TyreDegradationModel:
    """
    XGBoost Regressor to predict tyre life and pace drop-off.
    """
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            objective='reg:squarederror'
        )
        self.logger = logging.getLogger("ApexBrain_ML")
        self.is_trained = False

    def prepare_training_data(self, laps_data: pd.DataFrame):
        """
        Feature Engineering: Converts raw laps into ML-ready vectors.
        """
        # Filter for representative laps (clean laps only)
        # 1. Must be racing laps (not in/out laps)
        clean_laps = laps_data[
            (laps_data['PitOutTime'].isnull()) & 
            (laps_data['PitInTime'].isnull())
        ].copy()

        # 2. Remove Safety Car laps (outliers)
        # Heuristic: Remove laps > 107% of session best
        threshold = clean_laps['LapTime'].dt.total_seconds().min() * 1.07
        clean_laps = clean_laps[clean_laps['LapTime'].dt.total_seconds() < threshold]

        # 3. Feature Selection
        # We need to map strings (Compounds) to Integers or One-Hot
        clean_laps['Compound_Code'] = clean_laps['Compound'].map({'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}).fillna(1)
        
        # Features: [TyreAge, Compound, LapNumber (proxy for fuel)]
        X = clean_laps[['TyreLife', 'Compound_Code', 'LapNumber']]
        
        # Target: LapTime (in seconds)
        y = clean_laps['LapTime'].dt.total_seconds()
        
        return X, y

    def train(self, laps_data: pd.DataFrame):
        """
        Trains the model on the provided session data.
        """
        try:
            self.logger.info("Starting ML Training...")
            X, y = self.prepare_training_data(laps_data)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model.fit(X_train, y_train)
            
            preds = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            self.is_trained = True
            self.logger.info(f"Model Trained. RMSE: {rmse:.3f}s")
            return rmse
            
        except Exception as e:
            self.logger.error(f"Training Failed: {e}")
            return None

    def predict_wear_curve(self, compound: str, total_laps=50):
        """
        Generates a prediction curve for a specific compound over a stint.
        """
        if not self.is_trained:
            # Fallback to linear physics if model isn't trained
            return np.linspace(0, 2.0, total_laps)

        comp_code = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}.get(compound.upper(), 1)
        
        # Create a synthetic stint
        future_laps = pd.DataFrame({
            'TyreLife': np.arange(1, total_laps + 1),
            'Compound_Code': comp_code,
            'LapNumber': np.arange(1, total_laps + 1) # Assumes fresh race start
        })
        
        predicted_times = self.model.predict(future_laps)
        return predicted_times