import numpy as np
import pandas as pd
import itertools
import random
from dataclasses import dataclass
from typing import List, Dict

# ==========================================
# 1. DATA STRUCTURES
# ==========================================

@dataclass
class CircuitParams:
    name: str
    n_laps: int
    track_length_km: float
    base_lap_time: float
    pit_loss: float
    overtake_delta: float
    abrasivity: float

@dataclass
class CarParams:
    fuel_start_kg: float
    fuel_burn_kg_lap: float
    fuel_effect_s_kg: float
    pit_reliability_sigma: float

@dataclass
class EnvParams:
    track_temp: float
    rain_prob: float
    sc_prob: float
    vsc_prob: float
    n_drivers: int
    grid_position: int

@dataclass
class TireParams:
    compound: str
    base_pace: float
    deg_per_lap: float
    max_life: int
    warmup_loss: float

# ==========================================
# 2. THE MCDA ENGINE (FIXED)
# ==========================================

class F1DecisionEngine:
    
    def __init__(self, circuit: CircuitParams, car: CarParams, env: EnvParams, tires: Dict[str, TireParams], k_factors: Dict[str, float]):
        self.circuit = circuit
        self.car = car
        self.env = env
        self.tires = tires
        self.k_factors = k_factors

    # --- HELPER: STRATEGY STRING ---
    def format_strategy_string(self, strategy_stints):
        parts = []
        for s in strategy_stints:
            c = s['compound'][0]
            l = s['laps']
            # Icons
            if 'SOFT' in s['compound']: icon = "🔴"
            elif 'MEDIUM' in s['compound']: icon = "🟡"
            else: icon = "⚪"
            parts.append(f"{icon} {c}({l})")
        return " ➝ ".join(parts)

    # --- 1. CRITERIA CALCULATION ---

    def _calculate_c1_time(self, strategy) -> float:
        current_time = 0.0
        fuel = self.car.fuel_start_kg
        for idx, stint in enumerate(strategy):
            comp = stint['compound']
            n_laps = stint['laps']
            tire = self.tires[comp]
            
            if idx > 0: current_time += self.circuit.pit_loss
            current_time += tire.warmup_loss
            
            laps = np.arange(n_laps)
            t_fuel = (fuel - (laps * self.car.fuel_burn_kg_lap)) * self.car.fuel_effect_s_kg
            t_deg = laps * tire.deg_per_lap
            lap_times = self.circuit.base_lap_time + tire.base_pace + t_fuel + t_deg
            
            current_time += np.sum(lap_times)
            fuel -= (n_laps * self.car.fuel_burn_kg_lap)
        return current_time

    def _calculate_c2_safety_risk(self, strategy) -> float:
        risk_ratios = []
        for s in strategy:
            max_life = self.tires[s['compound']].max_life
            ratio = (s['laps'] / max_life) * 100
            risk_ratios.append(ratio)
        return max(risk_ratios)

    def _calculate_c3_traffic_score(self, strategy) -> float:
        n_stops = len(strategy) - 1
        grid_factor = self.env.grid_position / 20.0
        delta_factor = self.circuit.overtake_delta / 1.5
        raw_score = (grid_factor * 2.0) + (n_stops * grid_factor * delta_factor * 2.5)
        return min(10.0, raw_score)

    def _calculate_c4_flexibility_score(self, strategy) -> float:
        total_width = 0
        for stint in strategy:
            limit = self.tires[stint['compound']].max_life * 0.9
            width = max(0, limit - stint['laps'])
            total_width += width
        avg = total_width / len(strategy)
        return min(10.0, (avg / 15.0) * 10.0)

    # --- 2. DYNAMIC WEIGHTING ---

    def calculate_dynamic_weights(self) -> Dict[str, float]:
        w1 = 1.0 # Performance Anchor
        
        norm_temp = self.env.track_temp / 60.0
        norm_abr = self.circuit.abrasivity / 5.0
        w2 = self.k_factors['safety'] * norm_temp * norm_abr
        
        norm_grid = self.env.grid_position / 20.0
        w3 = self.k_factors['traffic'] * norm_grid
        
        chaos_sum = self.env.rain_prob + self.env.sc_prob + self.env.vsc_prob
        w4 = self.k_factors['robust'] * chaos_sum
        
        total_w = w1 + w2 + w3 + w4
        
        return {
            'alpha_1': w1 / total_w,
            'alpha_2': w2 / total_w,
            'alpha_3': w3 / total_w,
            'alpha_4': w4 / total_w
        }

    # --- 3. UTILITY & OPTIMIZER (FIXED) ---

    def calculate_utility(self, strategies):
        """
        Batch normalization and scoring.
        Ensures 'Normalized_Scores' key is added to every strategy.
        """
        if not strategies: return []
        weights = self.calculate_dynamic_weights()
        
        c1_raw = [s['C1_Time'] for s in strategies]
        min_t, max_t = min(c1_raw), max(c1_raw)
        range_t = max_t - min_t if max_t != min_t else 1.0
        
        results = []
        for s in strategies:
            # Normalize C1 (Time) locally [0-10]
            c1_score = ((s['C1_Time'] - min_t) / range_t) * 10.0
            
            # Normalize C2 (Risk) - Cap at 10 for score calculation
            c2_score = min(10.0, s['C2_Risk'] / 10.0)
            
            c3_score = s['C3_Traffic']
            c4_score = s['C4_Flex']
            
            # Weighted Sum Cost (Lower is Better)
            # Note: C4 is subtracted because it is a benefit
            utility = (weights['alpha_1'] * c1_score) + \
                      (weights['alpha_2'] * c2_score) + \
                      (weights['alpha_3'] * c3_score) - \
                      (weights['alpha_4'] * c4_score)
            
            s['Utility_Score'] = utility
            
            # CRITICAL FIX: Attach normalized breakdown for Radar Chart
            s['Normalized_Scores'] = {
                'C1': c1_score, 'C2': c2_score, 'C3': c3_score, 'C4': c4_score
            }
            results.append(s)
            
        return sorted(results, key=lambda x: x['Utility_Score'])

    def evaluate_strategy(self, strategy_def):
        c1 = self._calculate_c1_time(strategy_def)
        c2 = self._calculate_c2_safety_risk(strategy_def)
        c3 = self._calculate_c3_traffic_score(strategy_def)
        c4 = self._calculate_c4_flexibility_score(strategy_def)
        
        return {
            "Name": self.format_strategy_string(strategy_def),
            "C1_Time": c1,
            "C2_Risk": c2,
            "C3_Traffic": c3,
            "C4_Flex": c4,
            "Stints": strategy_def,
            # Placeholder until calculate_utility is run on the batch
            "Utility_Score": 0, 
            "Normalized_Scores": {'C1':0,'C2':0,'C3':0,'C4':0} 
        }

    def generate_optimal_strategies(self, n_gen=50):
        """
        Generates valid strategies and runs batch normalization.
        """
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        total_laps = self.circuit.n_laps
        candidates = []
        
        # 1. Generate 1-Stops
        for _ in range(n_gen // 2):
            pit_lap = random.randint(int(total_laps*0.25), int(total_laps*0.75))
            c1, c2 = random.sample(compounds, 2)
            
            # HARD CONSTRAINT: Tire Life Check
            l1, l2 = pit_lap, total_laps - pit_lap
            if l1 > self.tires[c1].max_life: continue
            if l2 > self.tires[c2].max_life: continue
            
            s = [{'compound': c1, 'laps': l1}, {'compound': c2, 'laps': l2}]
            candidates.append(self.evaluate_strategy(s))
            
        # 2. Generate 2-Stops
        for _ in range(n_gen // 2):
            l1 = int(total_laps/3) + random.randint(-5, 5)
            l2 = int(total_laps/3) + random.randint(-5, 5)
            l3 = total_laps - l1 - l2
            
            if l1 < 5 or l2 < 5 or l3 < 5: continue
            
            c_seq = [random.choice(compounds) for _ in range(3)]
            if len(set(c_seq)) < 2: continue # Must use 2 types
            
            # HARD CONSTRAINT
            if l1 > self.tires[c_seq[0]].max_life: continue
            if l2 > self.tires[c_seq[1]].max_life: continue
            if l3 > self.tires[c_seq[2]].max_life: continue
            
            s = [{'compound': c_seq[0], 'laps': l1}, 
                 {'compound': c_seq[1], 'laps': l2}, 
                 {'compound': c_seq[2], 'laps': l3}]
            candidates.append(self.evaluate_strategy(s))
        
        # CRITICAL FIX: Run normalization before returning
        ranked = self.calculate_utility(candidates)
        
        # Return unique top 5 based on name to avoid duplicates
        seen = set()
        unique = []
        for r in ranked:
            if r['Name'] not in seen:
                unique.append(r)
                seen.add(r['Name'])
                if len(unique) >= 5: break
                
        return unique