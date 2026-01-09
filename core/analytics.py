import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# ==========================================
# 1. PHYSICS ENGINE (The "CornerAnalyst")
# ==========================================
class CornerAnalyst:
    """
    Advanced Physics Engine for Telemetry Analysis.
    Handles: Corner Detection, Driver DNA, Overtake Physics, and Stint Analysis.
    """
    
    # --- A. CORNER DETECTION & ALIGNMENT ---
    def detect_corners(self, tel: pd.DataFrame):
        """
        Identifies corners using local minima in speed trace.
        Returns a DataFrame of corners with entry/exit physics data.
        """
        if tel.empty: return pd.DataFrame()
        
        speed = tel['Speed'].values
        dist = tel['Distance'].values
        
        # Find local minima (Apexes) - inverted peaks
        peaks, _ = find_peaks(-speed, distance=150, prominence=10)
        
        corners = []
        for i, idx in enumerate(peaks):
            apex_speed = speed[idx]
            apex_dist = dist[idx]
            
            # Braking Analysis: Min G_Long in 200m window before apex
            start = max(0, idx - 20)
            segment = tel.iloc[start:idx]
            braking = segment['G_Long'].min() if not segment.empty else 0
            
            corners.append({
                "Corner": f"T{i+1}",
                "Distance": apex_dist,
                "Apex_Speed": apex_speed,
                "Min_G_Brake": braking
            })
            
        return pd.DataFrame(corners)

    def align_corners(self, c1, c2):
        """
        Robustly matches corners between two drivers based on Track Position (Distance).
        Solves the 'KeyError' by ensuring we only compare corners that exist for both.
        """
        if c1.empty or c2.empty:
            return pd.DataFrame()

        aligned = []
        # Loop through Driver 1's corners
        for _, r1 in c1.iterrows():
            ref_dist = r1['Distance']
            
            # Find closest corner in Driver 2's data (Tolerance: 100m)
            c2['Dist_Diff'] = (c2['Distance'] - ref_dist).abs()
            match = c2.loc[c2['Dist_Diff'] < 100].sort_values('Dist_Diff')
            
            if not match.empty:
                r2 = match.iloc[0]
                delta = r1['Apex_Speed'] - r2['Apex_Speed']
                
                aligned.append({
                    "Corner": r1['Corner'],
                    "Distance": ref_dist,
                    "Driver1_Speed": r1['Apex_Speed'],
                    "Driver2_Speed": r2['Apex_Speed'],
                    "Apex_Delta": delta
                })
        
        return pd.DataFrame(aligned)

    # --- B. DRIVER DNA (RADAR CHART) ---
    def calculate_driver_radar(self, t1, t2):
        """
        Calculates 5-axis Driver DNA metrics for the Spider Chart.
        """
        def get_metrics(t):
            try:
                # Smoothness: Inverse of jerk (derivative of acceleration)
                jerk = t['Speed'].diff().diff().abs().mean()
                smoothness = 1 / (1 + jerk) if jerk > 0 else 1
                
                # Aggression: Variance in throttle application (stabbing the gas)
                aggression = t['Throttle'].diff().abs().mean()
                
                # Braking: Average Peak Braking Force
                braking = t.loc[t['Brake'] > 0, 'G_Long'].min() * -1 # Positive value
                
                # Cornering: Avg Lateral G in slow/med corners
                cornering = t.loc[t['Speed'] < 180, 'G_Lat'].abs().mean()
                
                return {
                    "Smoothness": min(smoothness * 8, 1.0),
                    "Aggression": min(aggression * 4, 1.0),
                    "Braking": min(braking / 5, 1.0),
                    "Cornering": min(cornering / 2.5, 1.0),
                    "Consistency": 0.85 # Placeholder for Stint consistency
                }
            except:
                return {"Smoothness":0, "Aggression":0, "Braking":0, "Cornering":0, "Consistency":0}

        return get_metrics(t1), get_metrics(t2)

    # --- C. BATTLE LOGIC (OVERTAKES & ANNOTATIONS) ---
    def analyze_overtake_probability(self, chaser_tel, leader_tel):
        """
        Calculates overtake probability based on Top Speed (DRS) and Traction Delta.
        """
        v1 = chaser_tel['Speed'].max()
        v2 = leader_tel['Speed'].max()
        delta_v = v1 - v2
        
        prob = 10
        reason = "Gap too stable"
        
        # Velocity Rules
        if delta_v > 15:
            prob += 75
            reason = "Massive Overspeed (+15kph)"
        elif delta_v > 5:
            prob += 30
            reason = "DRS Effective Range"
        elif delta_v < -5:
            prob -= 5
            reason = "Chaser hitting drag wall"
            
        # Traction Rules (Acceleration out of slow corners)
        # We look at G_Long when speed < 100kph
        trac_mask = (chaser_tel['Speed'] < 100) & (chaser_tel['G_Long'] > 0)
        if trac_mask.sum() > 10:
            acc_chaser = chaser_tel[trac_mask]['G_Long'].mean()
            acc_leader = leader_tel[trac_mask]['G_Long'].mean()
            if acc_chaser > acc_leader + 0.05:
                prob += 15
                reason += " + Superior Traction"
        
        return min(max(prob, 5), 95), reason

    def generate_plot_annotations(self, t1, t2, delta_trace):
        """
        Generates text arrows for charts highlighting the critical moment.
        """
        if delta_trace.empty: return []
        
        # Find the point of maximum time difference
        idx = delta_trace.abs().idxmax()
        dist = t1.loc[idx, 'Distance']
        val = delta_trace.loc[idx]
        
        return [{
            'x': dist, 
            'y': val, 
            'xref': 'x', 'yref': 'y',
            'text': f"<b>DECISIVE MOMENT</b><br>Gap: {abs(val):.2f}s",
            'showarrow': True, 
            'arrowhead': 2, 
            'arrowsize': 1,
            'arrowcolor': '#6366f1',
            'ax': 0, 
            'ay': -40,
            'bgcolor': '#ffffff',
            'bordercolor': '#6366f1',
            'borderpad': 4
        }]

    # --- D. STINT ANALYSIS (NEW FEATURE) ---
    def analyze_stint(self, session, driver):
        """
        Identifies and analyzes race stints (consecutive laps on same tyre).
        Used for the 'Stint Analysis' tab.
        """
        try:
            # Get valid laps with timing
            laps = session.laps.pick_driver(driver).pick_quicklaps().reset_index(drop=True)
            if laps.empty: return []
            
            # Identify Stint Changes (where compound changes)
            laps['Stint_ID'] = (laps['Compound'] != laps['Compound'].shift()).cumsum()
            
            stint_data = []
            for s_id in laps['Stint_ID'].unique():
                stint = laps[laps['Stint_ID'] == s_id]
                if len(stint) < 3: continue # Ignore in/out laps or tiny stints
                
                compound = stint['Compound'].iloc[0]
                avg_pace = stint['LapTime'].dt.total_seconds().mean()
                
                # Calculate Degradation (Slope of LapTime vs TyreAge)
                # Positive Slope = Degrading (Getting Slower)
                if len(stint) > 3:
                    deg = np.polyfit(stint['TyreLife'].astype(float), stint['LapTime'].dt.total_seconds(), 1)[0]
                else:
                    deg = 0.0
                
                stint_data.append({
                    "Stint_ID": int(s_id),
                    "Compound": compound,
                    "Laps_Count": len(stint),
                    "Start_Lap": stint['LapNumber'].min(),
                    "End_Lap": stint['LapNumber'].max(),
                    "Avg_Pace": avg_pace,
                    "Degradation": deg, # Seconds lost per lap
                    "Lap_Data": stint[['LapNumber', 'LapTime', 'TyreLife']].copy()
                })
                
            return stint_data
        except Exception as e:
            return []

    # --- E. CAR SETUP TRAITS (NEW FEATURE) ---
    def calculate_setup_traits(self, session):
        """
        Returns a DataFrame of Top Speed vs Cornering Speed for all drivers.
        Used to classify cars as 'High Downforce' or 'Low Drag'.
        """
        drivers = session.results['Abbreviation'].unique()
        traits = []
        
        for d in drivers:
            try:
                l = session.laps.pick_driver(d).pick_fastest()
                if l is None: continue
                t = l.get_telemetry()
                
                # V_Max: Max Speed
                v_max = t['Speed'].max()
                
                # Cornering: Min Speed in slow corners (approximate downforce proxy)
                # We filter for speeds between 60 and 120 kph to find slow corners
                slow_corners = t[(t['Speed'] > 60) & (t['Speed'] < 120)]
                v_min = slow_corners['Speed'].mean() if not slow_corners.empty else 0
                
                traits.append({
                    "Driver": d,
                    "Top_Speed": v_max,
                    "Cornering_Speed": v_min,
                    "Team": session.results.loc[session.results['Abbreviation']==d, 'TeamName'].iloc[0]
                })
            except:
                continue
                
        return pd.DataFrame(traits)
    
    def calculate_mini_sectors(self, t1, t2, n_sectors=25):
        """
        Divides the track into N mini-sectors and identifies the faster driver in each.
        Returns a DataFrame for the Track Map visualization.
        """
        # Create bins based on distance
        max_dist = max(t1['Distance'].max(), t2['Distance'].max())
        bins = np.linspace(0, max_dist, n_sectors + 1)
        
        # Assign sector IDs to telemetry
        t1['Sector_ID'] = pd.cut(t1['Distance'], bins, labels=False)
        t2['Sector_ID'] = pd.cut(t2['Distance'], bins, labels=False)
        
        sector_data = []
        
        for i in range(n_sectors):
            # Get data for this chunk
            c1 = t1[t1['Sector_ID'] == i]
            c2 = t2[t2['Sector_ID'] == i]
            
            if c1.empty or c2.empty: continue
            
            # Calculate mean speed
            v1 = c1['Speed'].mean()
            v2 = c2['Speed'].mean()
            
            # Determine winner
            diff = v1 - v2
            winner = 1 if diff > 0 else 2 # 1=Driver1, 2=Driver2
            
            # Get coordinates for plotting (center of sector)
            # We take the mean X/Y of this chunk
            avg_x = c1['X'].mean()
            avg_y = c1['Y'].mean()
            
            sector_data.append({
                "Sector": i,
                "X": avg_x,
                "Y": avg_y,
                "Winner": winner,
                "Delta": diff,
                "Speed_D1": v1,
                "Speed_D2": v2
            })
            
        return pd.DataFrame(sector_data)

    def analyze_traction(self, t1, t2):
        """
        Returns data for Traction Analysis (Speed vs Throttle).
        Filters for 'Corner Exit' phases (Low Speed + Increasing Throttle).
        """
        # Filter: Speed < 160kph AND Throttle > 0
        mask1 = (t1['Speed'] < 160) & (t1['Throttle'] > 10)
        mask2 = (t2['Speed'] < 160) & (t2['Throttle'] > 10)
        
        return t1[mask1], t2[mask2]
    
    def calculate_corner_types(self, t1, t2):
        """
        Groups corners into Low, Medium, and High speed buckets and calculates the delta.
        Returns a summary DataFrame.
        """
        # Detect corners for both
        c1 = self.detect_corners(t1)
        c2 = self.detect_corners(t2)
        aligned = self.align_corners(c1, c2)
        
        if aligned.empty: return pd.DataFrame()
        
        # Categorize Corners
        # Low < 100kph | Med 100-180kph | High > 180kph
        def categorize(speed):
            if speed < 100: return "Low Speed (<100)"
            if speed < 180: return "Med Speed (100-180)"
            return "High Speed (>180)"
        
        aligned['Type'] = aligned['Driver1_Speed'].apply(categorize)
        
        # Calculate Average Delta per Type
        summary = aligned.groupby('Type')['Apex_Delta'].mean().reset_index()
        summary['Count'] = aligned.groupby('Type')['Corner'].count().values
        
        return summary

    def calculate_ideal_lap(self, session, driver):
        """
        Constructs the Theoretical Best Lap from best sectors across all laps.
        """
        laps = session.laps.pick_driver(driver).pick_quicklaps()
        if laps.empty: return None
        
        best_s1 = laps['Sector1Time'].min()
        best_s2 = laps['Sector2Time'].min()
        best_s3 = laps['Sector3Time'].min()
        
        theoretical = best_s1 + best_s2 + best_s3
        actual_best = laps['LapTime'].min()
        
        return {
            "Driver": driver,
            "Best_S1": best_s1.total_seconds(),
            "Best_S2": best_s2.total_seconds(),
            "Best_S3": best_s3.total_seconds(),
            "Theoretical_Lap": theoretical.total_seconds(),
            "Actual_Lap": actual_best.total_seconds(),
            "Time_Left_On_Table": (actual_best - theoretical).total_seconds()
        }
    
    
    

# ==========================================
# 2. HISTORICAL VAULT (Season Analysis)
# ==========================================
class HistoricalVault:
    """
    Longitudinal Season Analysis.
    Compares team performance evolution over the year.
    """
    
    def calculate_gap_to_pole(self, session_results):
        """
        Calculates the % performance deficit relative to Pole Position.
        """
        if not hasattr(session_results, 'empty') or session_results.empty:
            return None
            
        # Get Pole Time
        valid_res = session_results[session_results['Time'].notna()].sort_values('Time')
        if valid_res.empty: return None
        
        pole_time = valid_res.iloc[0]['Time'].total_seconds()
        
        team_stats = []
        for team in session_results['TeamName'].unique():
            team_drivers = session_results[session_results['TeamName'] == team]
            if team_drivers.empty: continue
            
            best_lap = team_drivers['Time'].min().total_seconds()
            gap_pct = ((best_lap / pole_time) - 1) * 100
            
            team_stats.append({
                "Team": team,
                "Gap_Pct": gap_pct
            })
            
        return pd.DataFrame(team_stats).sort_values("Gap_Pct")