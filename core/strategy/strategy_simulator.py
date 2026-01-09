# core/strategy/strategy_simulator.py

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import List, Dict

# Setup Logger
logger = logging.getLogger("ApexBrain_Strategy")

@dataclass
class CompoundProfile:
    """Defines the physics profile of a tyre compound."""
    name: str          # "Soft", "Medium", "Hard"
    base_pace: float   # Relative pace offset (e.g., Soft=0.0, Hard=0.8)
    deg_per_lap: float # Degradation coefficient
    max_life: int      # Theoretical structural limit

class StrategyOracle:
    """
    Monte Carlo Simulation Engine for F1 Race Strategy.
    """
    
    def __init__(self):
        # Default Physics Constants (tuned for 2024/25 cars)
        self.FUEL_CORRECTION = 0.035  # Seconds gained per lap of fuel burn
        self.PIT_LOSS = 22.5          # Average pit lane loss (track dependent)
        
        # Compound Profiles (Ideally loaded from ML models)
        self.compounds = {
            "SOFT": CompoundProfile("Soft", base_pace=0.0, deg_per_lap=0.12, max_life=20),
            "MEDIUM": CompoundProfile("Medium", base_pace=0.6, deg_per_lap=0.08, max_life=35),
            "HARD": CompoundProfile("Hard", base_pace=1.1, deg_per_lap=0.04, max_life=55)
        }

    def simulate_stint(self, compound: str, laps: int, start_tyre_age: int = 0, driver_sigma: float = 0.2):
        """
        Simulates a single stint and returns array of lap times.
        """
        c = self.compounds.get(compound.upper())
        if not c:
            raise ValueError(f"Unknown Compound: {compound}")

        # Vectorized Lap Generation
        lap_indices = np.arange(laps)
        tyre_age = lap_indices + start_tyre_age
        
        # 1. Degradation (Non-linear: wear accelerates near end of life)
        deg_penalty = c.deg_per_lap * (tyre_age ** 1.3)
        
        # 2. Fuel Effect (Linear gain)
        # Note: This simplifies fuel load; assumes race starts at lap 0 logic
        fuel_gain = self.FUEL_CORRECTION * lap_indices 
        
        # 3. Base Pace + Compound Offset
        base_times = np.full(laps, 90.0 + c.base_pace) # 90.0s arbitrary baseline
        
        # 4. Random Noise (Driver Consistency)
        noise = np.random.normal(0, driver_sigma, laps)
        
        # Calculate final lap times
        stint_times = base_times + deg_penalty - fuel_gain + noise
        
        return stint_times

    def run_strategy(self, strategy_plan: List[tuple], total_laps: int):
        """
        Runs one full race simulation based on a strategy plan.
        strategy_plan format: [('SOFT', 15), ('HARD', 40)] -> (Compound, Laps to run)
        """
        race_times = []
        cumulative_time = 0
        current_lap = 0
        
        for i, (compound, stint_length) in enumerate(strategy_plan):
            # Enforce race distance limit
            if current_lap + stint_length > total_laps:
                stint_length = total_laps - current_lap
            
            # Simulate Stint
            stint_times = self.simulate_stint(compound, stint_length)
            
            # Add Pit Loss (except for race start)
            if i > 0:
                cumulative_time += self.PIT_LOSS
                # Add pit loss to the first lap of the new stint for visualization
                stint_times[0] += self.PIT_LOSS 
            
            race_times.extend(stint_times)
            cumulative_time += np.sum(stint_times)
            current_lap += stint_length
            
            if current_lap >= total_laps:
                break
                
        return np.array(race_times), cumulative_time

    def monte_carlo_simulation(self, strategies: Dict[str, List[tuple]], n_sims=1000, total_laps=57):
        """
        Runs N simulations for different strategies to find the winner.
        """
        results = {}
        
        for strat_name, plan in strategies.items():
            sim_totals = []
            
            for _ in range(n_sims):
                _, total_time = self.run_strategy(plan, total_laps)
                sim_totals.append(total_time)
            
            sim_totals = np.array(sim_totals)
            
            results[strat_name] = {
                "mean_time": np.mean(sim_totals),
                "p25": np.percentile(sim_totals, 25), # Best case
                "p75": np.percentile(sim_totals, 75), # Worst case
                "std_dev": np.std(sim_totals),
                "win_prob": 0 # To be calculated relative to others
            }
            
        # Calculate Win Probability (Simple comparison of means for now)
        # In production, we would compare distribution overlaps.
        best_strat = min(results, key=lambda x: results[x]['mean_time'])
        results[best_strat]['is_recommended'] = True
        
        return results

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    oracle = StrategyOracle()
    
    # Define two strategies for a 57 lap race (e.g., Bahrain)
    strategies = {
        "1-Stop (M-H)": [('MEDIUM', 25), ('HARD', 32)],
        "2-Stop (S-M-M)": [('SOFT', 15), ('MEDIUM', 21), ('MEDIUM', 21)]
    }
    
    print("Running Monte Carlo Simulation (n=2000)...")
    results = oracle.monte_carlo_simulation(strategies, n_sims=2000)
    
    for name, metrics in results.items():
        print(f"Strategy: {name}")
        print(f"  Avg Race Time: {metrics['mean_time']:.2f}s")
        print(f"  Risk (StdDev): {metrics['std_dev']:.2f}s")
        if metrics.get('is_recommended'):
            print("  >> RECOMMENDED STRATEGY <<")
        print("-" * 30)