import pandas as pd
import numpy as np
import fastf1

class StrategyConfig:
    def __init__(self, name, pit_laps, compounds):
        self.name = name
        self.pit_laps = pit_laps
        self.compounds = compounds

class RaceSimulator:
    def __init__(self, ml_model):
        self.ml_model = ml_model

    def simulate_race(self, config):
        # Placeholder for Monte Carlo logic if needed
        return {'mean': 0, 'raw_data': []}

class TrafficOracle:
    """
    Analyzes live traffic, gaps, and tyre history.
    """
    
    def get_race_trace(self, session):
        """
        Calculates the Gap to Leader for every driver on every lap.
        Returns a DataFrame for plotting.
        """
        if session.laps.empty: return None
        
        # Calculate Gap to Leader
        # We assume 'Time' is the cumulative race time
        laps = session.laps.pick_quicklaps() # Filter out bad data
        
        # Get the leader's time per lap
        leader = laps.pick_driver(session.results.iloc[0]['Abbreviation'])
        if leader.empty: return None
        
        # Resample logic simplified for visualization
        # In a real app, we'd use fastf1.core.Laps.pick_wo_box() alignment
        # Here we just use the pre-calculated 'GapToLeader' if available, or approx
        
        data = []
        for drv in session.results['Abbreviation']:
            d_laps = session.laps.pick_driver(drv)
            if d_laps.empty: continue
            
            # Simple cumulative time comparison vs Leader
            # (Note: Exact gap calculation is complex, this is a visual approximation)
            for i, row in d_laps.iterrows():
                try:
                    leader_lap = leader[leader['LapNumber'] == row['LapNumber']]
                    if not leader_lap.empty:
                        gap = row['Time'] - leader_lap.iloc[0]['Time']
                        data.append({
                            'Driver': drv,
                            'LapNumber': row['LapNumber'],
                            'GapToLeader': gap.total_seconds()
                        })
                except: continue
                
        return pd.DataFrame(data)

    def get_tyre_strategy_map(self, session):
        """
        Returns a DataFrame of all stints for all drivers.
        Format: [Driver, Stint_ID, Compound, StartLap, EndLap, Laps]
        """
        stints = []
        for drv in session.results['Abbreviation']:
            d_laps = session.laps.pick_driver(drv)
            if d_laps.empty: continue
            
            # Identify stint changes
            d_laps['Stint'] = d_laps['Stint'].fillna(1).astype(int)
            
            for s_id in d_laps['Stint'].unique():
                stint_laps = d_laps[d_laps['Stint'] == s_id]
                if stint_laps.empty: continue
                
                compound = stint_laps.iloc[0]['Compound']
                start = stint_laps['LapNumber'].min()
                end = stint_laps['LapNumber'].max()
                length = len(stint_laps)
                
                stints.append({
                    'Driver': drv,
                    'Stint': s_id,
                    'Compound': compound,
                    'StartLap': start,
                    'EndLap': end,
                    'Laps': length
                })
        
        return pd.DataFrame(stints)

    def calculate_pit_rejoin(self, session, driver_code, pit_loss=22.0):
        """
        Predicts traffic after a pit stop.
        pit_loss: Time lost in pit lane (seconds).
        Returns DataFrame of drivers who will be near the rejoin point.
        """
        # 1. Get Driver's current position/time
        # We use the latest completed lap data
        last_lap_n = session.laps['LapNumber'].max()
        
        # Get everyone's total race time at the last lap
        current_state = []
        for drv in session.results['Abbreviation']:
            d_laps = session.laps.pick_driver(drv)
            if d_laps.empty: continue
            
            # Get last lap time
            last = d_laps.iloc[-1]
            if pd.isna(last['Time']): continue
            
            current_state.append({
                'Driver': drv,
                'RaceTime': last['Time'].total_seconds(),
                'Lap': last['LapNumber']
            })
            
        df = pd.DataFrame(current_state)
        if df.empty or driver_code not in df['Driver'].values: return pd.DataFrame()
        
        # 2. Calculate Driver's Rejoin Time
        my_time = df[df['Driver'] == driver_code].iloc[0]['RaceTime']
        rejoin_time = my_time + pit_loss
        
        # 3. Find threats (Drivers who have a RaceTime close to rejoin_time)
        # Threat = RaceTime is less than Rejoin Time (Ahead) but close
        # Or RaceTime is slightly more (Behind)
        
        df['GapToRejoin'] = df['RaceTime'] - rejoin_time
        
        # Filter for relevant cars (e.g. +/- 5 seconds window)
        # Negative Gap = They are Ahead of us
        # Positive Gap = They are Behind us
        window = df[ (df['GapToRejoin'] > -5) & (df['GapToRejoin'] < 10) ].copy()
        
        # Sort by Gap (Those ahead first)
        window = window.sort_values('GapToRejoin')
        
        return window[['Driver', 'GapToRejoin']]