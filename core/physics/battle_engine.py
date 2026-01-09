# core/physics/battle_engine.py

import numpy as np
import pandas as pd
import fastf1
from fastf1 import utils

class BattleEngine:
    """
    Handles Head-to-Head comparisons and Delta calculations.
    """
    
    def calculate_delta(self, lap_ref, lap_target):
        """
        Calculates the time delta between a Reference Lap and a Target Lap.
        
        Args:
            lap_ref (Lap): The baseline lap (usually the faster one).
            lap_target (Lap): The lap to compare against.
            
        Returns:
            pd.DataFrame: Merged telemetry with 'Delta' column.
        """
        # Get Telemetry
        ref_tel = lap_ref.get_telemetry()
        target_tel = lap_target.get_telemetry()
        
        # Ensure 'Distance' and 'Time' exist
        if 'Distance' not in ref_tel.columns or 'Distance' not in target_tel.columns:
            return None

        # 1. Create a common distance axis (using Reference's distance)
        # We perform interpolation to estimate Target's time at Ref's distance markers
        ref_dist = ref_tel['Distance']
        ref_time = ref_tel['Time'].dt.total_seconds()
        
        target_dist = target_tel['Distance']
        target_time = target_tel['Time'].dt.total_seconds()
        
        # Interpolate Target Time onto Reference Distance
        target_time_interp = np.interp(ref_dist, target_dist, target_time)
        
        # 2. Calculate Delta
        # Delta = Target Time - Ref Time
        # Positive Delta = Target is Slower (arrived later)
        delta = target_time_interp - ref_time
        
        # 3. Merge into a single DataFrame for plotting
        battle_df = pd.DataFrame({
            "Distance": ref_dist,
            "Ref_Speed": ref_tel['Speed'],
            "Target_Speed": np.interp(ref_dist, target_dist, target_tel['Speed']),
            "Delta": delta,
            # For 3D Map, we need X, Y coordinates
            "X": ref_tel['X'],
            "Y": ref_tel['Y'],
            "Z": ref_tel['Z']
        })
        
        return battle_df

    def get_mini_sectors(self, battle_df, n_sectors=20):
        """
        Aggregates delta into mini-sectors to see WHERE time was gained.
        """
        battle_df['MiniSector'] = pd.cut(battle_df['Distance'], bins=n_sectors, labels=False)
        sector_analysis = battle_df.groupby('MiniSector')[['Delta']].last().diff().fillna(0)
        return sector_analysis