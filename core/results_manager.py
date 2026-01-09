import pandas as pd
import numpy as np
import fastf1

class ResultsManager:
    def get_results(self, session):
        """Returns a cleaned results table, handling NaNs safely."""
        if session.results.empty: return None
        df = session.results.copy()
        
        # 1. Safe Position Filling
        if 'Position' in df.columns:
            safe_ranks = np.arange(1, len(df) + 1)
            df['Position'] = df['Position'].fillna(pd.Series(safe_ranks, index=df.index))
            try:
                df['Position'] = df['Position'].astype(int).astype(str)
            except:
                df['Position'] = df['Position'].astype(str).str.replace('.0', '', regex=False)

        # 2. Gap Calculation
        if 'Time' in df.columns:
            leader_time = df.iloc[0]['Time']
            df['GapToLeader'] = (df['Time'] - leader_time).dt.total_seconds()
            df['GapToLeader'] = df['GapToLeader'].fillna(0).apply(lambda x: f"+{x:.1f}s" if x > 0 else "LEADER")
            
        return df[['Position', 'Abbreviation', 'TeamName', 'GapToLeader', 'Time']]

    def get_fastest_lap_comparison(self, session):
        try:
            laps = session.laps.pick_quicklaps()
            return pd.DataFrame([laps.pick_fastest()])
        except: return None

    def generate_replay_frame(self, session):
        """
        Generates 0-100% replay data.
        FIXED: Prevents 'Driver' column from disappearing during fill operations.
        """
        if session.laps.empty: return None
        
        # 1. Collect Raw Telemetry
        raw_frames = []
        try:
            fastest = session.laps.pick_fastest()
            # Fallback if fastest is None
            if fastest is None: fastest = session.laps.pick_wo_box().iloc[0]
            max_dist = fastest.get_telemetry()['Distance'].max()
        except:
            max_dist = 5000 
            
        drivers = session.results['Abbreviation'].unique()
        
        for drv in drivers:
            try:
                d_laps = session.laps.pick_driver(drv).pick_quicklaps()
                if d_laps.empty: continue
                
                tel = d_laps.pick_fastest().get_telemetry()
                tel['Step'] = ((tel['Distance'] / max_dist) * 100).astype(int)
                
                tel = tel.groupby('Step')[['X', 'Y', 'Distance', 'Speed']].first().reset_index()
                tel['Driver'] = drv
                tel['Team'] = session.results.loc[session.results['Abbreviation'] == drv, 'TeamName'].values[0]
                
                raw_frames.append(tel)
            except: continue
            
        if not raw_frames: return None
        
        df_raw = pd.concat(raw_frames)
        
        # 2. Reindex to ensure every driver has steps 0-100
        steps = np.arange(0, 101)
        multi_index = pd.MultiIndex.from_product([drivers, steps], names=['Driver', 'Step'])
        
        # Merge raw data onto the perfect grid
        df_full = df_raw.set_index(['Driver', 'Step']).reindex(multi_index).reset_index()
        
        # 3. CRITICAL FIX: Fill Missing Data WITHOUT dropping 'Driver' column
        # We only apply ffill/bfill to the specific data columns, grouped by Driver
        cols_to_fill = ['X', 'Y', 'Distance', 'Speed', 'Team']
        
        # This fills the columns in place while respecting the group
        df_full[cols_to_fill] = df_full.groupby('Driver')[cols_to_fill].ffill().bfill()
        
        # 4. Restore Team Names Map (Safety net for pure NaNs)
        team_map = session.results.set_index('Abbreviation')['TeamName'].to_dict()
        
        # Now 'Driver' column is guaranteed to exist
        df_full['Team'] = df_full['Driver'].map(team_map)
        
        return df_full