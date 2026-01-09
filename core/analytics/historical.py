# core/analytics/historical.py

import pandas as pd
import numpy as np

class HistoricalVault:
    """
    Manages longitudinal data analysis across the season.
    """
    
    def get_season_development_trends(self, year=2024):
        """
        Generates the 'Aero Development War' dataset.
        In production, this queries a database of processed race results.
        Here, we generate a representative dataset mirroring real 2024 trends.
        """
        
        # Races (X-Axis)
        races = ["BHR", "SAU", "AUS", "JPN", "CHN", "MIA", "EMI", "MON", "CAN", "ESP"]
        
        # Trends (Y-Axis: % Gap to Pole)
        # 0.00 = Pole Position Pace
        
        # Red Bull: Starts dominant, slight regression
        rbr_trend = [0.00, 0.05, 0.00, 0.00, 0.00, 0.10, 0.05, 0.15, 0.05, 0.00]
        
        # McLaren: Starts slow, massive upgrade package at Miami (Index 5)
        mcl_trend = [0.80, 0.75, 0.60, 0.55, 0.50, 0.15, 0.10, 0.05, 0.00, 0.05]
        
        # Ferrari: Consistent challenger
        fer_trend = [0.30, 0.25, 0.10, 0.35, 0.40, 0.30, 0.25, 0.00, 0.20, 0.25]
        
        # Mercedes: Struggling early, slow convergence
        mer_trend = [0.60, 0.65, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25]

        data = {
            "Race": races,
            "Red Bull Racing": rbr_trend,
            "McLaren": mcl_trend,
            "Ferrari": fer_trend,
            "Mercedes": mer_trend
        }
        
        return pd.DataFrame(data)

    def calculate_consistency_score(self, driver_laps):
        """
        Calculates lap time variance (standard deviation) excluding outliers (Box-in/out).
        """
        # Filter for valid flying laps (within 107% of best)
        best_time = driver_laps.pick_fastest()['LapTime'].total_seconds()
        threshold = best_time * 1.07
        
        valid_laps = driver_laps[driver_laps['LapTime'].dt.total_seconds() < threshold]
        
        # Calculate consistency (lower std_dev is better)
        times = valid_laps['LapTime'].dt.total_seconds()
        return np.std(times)