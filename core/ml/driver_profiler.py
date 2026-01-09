# core/ml/driver_profiler.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class DriverProfiler:
    """
    Unsupervised Learning to cluster driver styles.
    """
    
    def analyze_styles(self, session):
        """
        Extracts telemetry signatures from all drivers in a session
        and clusters them.
        """
        driver_stats = []
        
        for driver in session.drivers:
            try:
                # Get fastest lap telemetry
                laps = session.laps.pick_driver(driver)
                if laps.empty: continue
                
                fastest = laps.pick_fastest()
                tel = fastest.get_telemetry()
                
                # Extract Features
                # 1. Aggressiveness (std dev of throttle derivative)
                throttle_agg = np.std(np.gradient(tel['Throttle']))
                
                # 2. Braking Late (max longitudinal G)
                # Note: FastF1 G-force is approximate, but usable for relative clustering
                if 'LinearAcceleration' in tel.columns:
                     brake_agg = abs(tel['LinearAcceleration'].min())
                else:
                     # Fallback derivation
                     v = tel['Speed'] / 3.6
                     brake_agg = abs(np.min(np.gradient(v)))

                driver_stats.append({
                    'Driver': driver,
                    'Throttle_Aggression': throttle_agg,
                    'Braking_Force': brake_agg
                })
                
            except Exception:
                continue
                
        # Create DataFrame
        df = pd.DataFrame(driver_stats)
        if df.empty: return None
        
        # Normalize Data
        scaler = StandardScaler()
        features = ['Throttle_Aggression', 'Braking_Force']
        X = scaler.fit_transform(df[features])
        
        # K-Means Clustering (k=3: Smooth, Balanced, Aggressive)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Map Clusters to Names (Heuristic based on centroids)
        # We need to see which cluster has the highest Aggression score
        centroids = kmeans.cluster_centers_
        # Sum of features for each centroid to gauge "Intensity"
        intensity = centroids.sum(axis=1)
        
        # Sort clusters by intensity
        mapping = {
            np.argsort(intensity)[0]: "Smooth / Preserver",
            np.argsort(intensity)[1]: "Balanced",
            np.argsort(intensity)[2]: "Aggressive / Late Braker"
        }
        
        df['Style'] = df['Cluster'].map(mapping)
        
        return df