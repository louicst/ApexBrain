import numpy as np
import pandas as pd
import itertools

class StrategyMCDA:
    """
    Multi-Criteria Decision Analysis Engine for F1 Strategy.
    """
    
    def __init__(self):
        # Default tyre specs (Degradation in s/lap, Max Life in laps)
        self.tyre_specs = {
            'SOFT': {'deg': 0.12, 'life': 20, 'pace_advantage': 0.0},
            'MEDIUM': {'deg': 0.08, 'life': 30, 'pace_advantage': 0.5},
            'HARD': {'deg': 0.04, 'life': 45, 'pace_advantage': 1.1}
        }

    def calculate_metrics(self, stints, env_vars):
        """
        Calculates C1 (Time), C2 (Safety), C3 (Traffic), C4 (Flexibility)
        """
        fuel_load = 110.0
        fuel_burn = 110.0 / env_vars['total_laps']
        current_time = 0.0
        
        lap_times = []
        c2_safety_scores = []
        
        for stint in stints:
            comp = stint['compound']
            n_laps = stint['laps']
            spec = self.tyre_specs[comp]
            
            # Pit Stop Cost (except for start)
            if len(lap_times) > 0:
                current_time += env_vars['pit_cost']
            
            for lap_in_stint in range(n_laps):
                fuel_penalty = fuel_load * env_vars['fuel_effect']
                tire_penalty = lap_in_stint * spec['deg']
                lap_time = env_vars['base_time'] + spec['pace_advantage'] + fuel_penalty + tire_penalty
                
                lap_times.append(lap_time)
                current_time += lap_time
                fuel_load -= fuel_burn
            
            # C2: Safety Risk % (Max life usage)
            risk = (n_laps / spec['life']) * 100
            c2_safety_scores.append(risk)

        # C3: Traffic (Heuristic)
        stops = len(stints) - 1
        traffic = min(10, (env_vars['grid_pos'] * 0.1) + (stops * 1.2))

        # C4: Flexibility (Reverse metric: Higher is better)
        # Score 0-10. Start at 10, deduct for Softs (short window) or many stops.
        flex = 10 - (stops * 2) - (2 if stints[0]['compound'] == 'SOFT' else 0)
        
        return {
            "TotalTime": current_time,
            "MaxRisk": max(c2_safety_scores) if c2_safety_scores else 0,
            "TrafficScore": traffic,
            "Flexibility": max(0, flex),
            "LapTrace": lap_times,
            "StintDef": stints
        }

    def find_optimal_strategy(self, env_vars):
        """
        Auto-Solver with SAFETY CONSTRAINT (Risk < 100%).
        """
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        laps = env_vars['total_laps']
        valid_strategies = []
        
        # 1-STOP Generator
        for p1 in range(int(laps*0.2), int(laps*0.8)):
            stint1, stint2 = p1, laps - p1
            for c1, c2 in itertools.product(compounds, repeat=2):
                if c1 == c2 and env_vars['require_compound_change']: continue
                
                s = [{'compound': c1, 'laps': stint1}, {'compound': c2, 'laps': stint2}]
                metrics = self.calculate_metrics(s, env_vars)
                
                # [FIX] STRICT SAFETY CHECK: Discard if risk > 98%
                if metrics['MaxRisk'] < 98.0:
                    valid_strategies.append({**metrics, "Name": f"AI 1-Stop ({c1[0]}-{c2[0]})"})

        # 2-STOP Generator (Simplified: Equal-ish stints)
        for c1, c2, c3 in itertools.product(compounds, repeat=3):
            # Try splits: 33/33/33
            s_len = int(laps / 3)
            s = [
                {'compound': c1, 'laps': s_len},
                {'compound': c2, 'laps': s_len},
                {'compound': c3, 'laps': laps - (s_len*2)}
            ]
            metrics = self.calculate_metrics(s, env_vars)
            
            if metrics['MaxRisk'] < 98.0:
                 valid_strategies.append({**metrics, "Name": f"AI 2-Stop ({c1[0]}-{c2[0]}-{c3[0]})"})

        # Sort by Time (Lowest is best)
        if not valid_strategies: return None
        valid_strategies.sort(key=lambda x: x['TotalTime'])
        
        return valid_strategies[0]
    
    def normalize_for_radar(self, strategies):
        """
        Normalizes metrics to 0-10 scale for Radar Chart.
        Higher is ALWAYS better in this view.
        """
        data = []
        
        # Find global min/max for normalization
        all_times = [s['TotalTime'] for s in strategies]
        min_time, max_time = min(all_times), max(all_times)
        time_range = max_time - min_time if max_time != min_time else 1

        for s in strategies:
            # 1. Performance: Invert (Fastest = 10, Slowest = 0)
            perf = 10 - ((s['TotalTime'] - min_time) / time_range * 10)
            
            # 2. Safety: Invert Risk (0% Risk = 10, 100% Risk = 0)
            safe = 10 - (min(s['MaxRisk'], 100) / 10)
            
            # 3. Traffic: Invert (0 Traffic = 10)
            traf = 10 - s['TrafficScore']
            
            # 4. Flexibility: Already 0-10 (Higher is better)
            flex = s['Flexibility']
            
            data.append({
                "Name": s['Name'],
                "Performance": perf,
                "Safety": safe,
                "CleanAir": traf,
                "Flexibility": flex
            })
            
        return pd.DataFrame(data)