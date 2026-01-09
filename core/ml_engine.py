import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ApexML:
    def __init__(self):
        self.deg_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1
        )
        self.classifier = KMeans(n_clusters=3, n_init=10) # 3 Clusters: Push, Race, Slow
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.is_classified = False

    def train_deg_model(self, laps):
        """
        Trains the Tyre Degradation Model (Supervised).
        """
        # (Keep existing logic for degradation training)
        df = laps.pick_quicklaps().reset_index(drop=True)
        if df.empty: return
        
        # Features: LapNumber, TyreLife, Compound_ID
        df['Comp_ID'] = df['Compound'].map({'SOFT':0, 'MEDIUM':1, 'HARD':2}).fillna(1)
        
        X = df[['TyreLife', 'Comp_ID', 'LapNumber']]
        y = df['LapTime'].dt.total_seconds()
        
        self.deg_model.fit(X, y)
        self.is_fitted = True

    def cluster_laps(self, laps):
        """
        Classifies ALL laps into [Push, HighFuel, Slow] using Unsupervised ML.
        """
        # Prepare Data: Filter out In/Out laps roughly
        df = laps.dropna(subset=['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']).copy()
        
        # Features for Clustering: LapTime, Sector Variance
        # We convert Timedeltas to Seconds
        df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
        df['S1'] = df['Sector1Time'].dt.total_seconds()
        df['S3'] = df['Sector3Time'].dt.total_seconds()
        
        # Normalize
        X = df[['LapTimeSec', 'S1', 'S3']]
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        df['Cluster_ID'] = self.classifier.fit_predict(X_scaled)
        
        # Label Clusters Logic:
        # The cluster with the Lowest Average LapTime is "PUSH"
        # The cluster with the Highest Average LapTime is "COOL/SLOW"
        stats = df.groupby('Cluster_ID')['LapTimeSec'].mean().sort_values()
        
        label_map = {
            stats.index[0]: '🔥 PUSH',      # Fastest
            stats.index[1]: '⚖️ RACE PACE', # Middle
            stats.index[2]: '🐢 COOL/SLOW'  # Slowest
        }
        
        df['Lap_Type'] = df['Cluster_ID'].map(label_map)
        self.is_classified = True
        
        return df