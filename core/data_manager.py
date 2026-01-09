# core/data_manager.py

import fastf1
import pandas as pd
import numpy as np
import logging
import streamlit as st
from datetime import timedelta

# SETUP LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    """
    ENTERPRISE DATA LAYER.
    Handles caching, session loading, and data normalization for ANY race/driver.
    """
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        fastf1.Cache.enable_cache(cache_dir)
        self.session = None
        self.laps = None
        self.telemetry_cache = {} # In-memory cache for speed

    def load_session(self, year, gp, session_type):
        """
        Loads full session data including weather and lap timing.
        """
        try:
            # Clear previous session data to free memory
            self.session = None 
            self.laps = None
            self.telemetry_cache = {}

            session = fastf1.get_session(year, gp, session_type)
            session.load(telemetry=True, weather=True, messages=True)
            
            self.session = session
            self.laps = session.laps
            
            # PRO FEATURE: Pre-calculate basic lap metrics for the whole grid
            # This makes the UI snappy later
            self.laps['LapTimeSec'] = self.laps['LapTime'].dt.total_seconds()
            
            logging.info(f"✅ Session Loaded: {year} {gp} {session_type}")
            return True, f"Session Loaded: {session.event['EventName']} - {session.name}"
            
        except Exception as e:
            logging.error(f"❌ Failed to load session: {e}")
            return False, str(e)

    def get_driver_list(self):
        """Dynamic list of drivers for the current session."""
        if self.session is None: return []
        # Return list of (Abbreviation, Full Name) tuples
        drivers = self.session.results
        return list(zip(drivers['Abbreviation'], drivers['FullName']))

    def get_clean_telemetry(self, driver, lap_number=None):
        """
        Robust telemetry fetcher. Handles missing data/crashes.
        Returns synchronized Distance/Speed/Time data.
        """
        cache_key = f"{driver}_{lap_number}"
        if cache_key in self.telemetry_cache:
            return self.telemetry_cache[cache_key]

        try:
            # Select Driver Laps
            d_laps = self.laps.pick_driver(driver)
            
            # Select Specific Lap or Fastest Lap
            if lap_number:
                lap = d_laps[d_laps['LapNumber'] == lap_number].iloc[0]
            else:
                lap = d_laps.pick_fastest()

            # Load Telemetry
            tel = lap.get_telemetry()
            
            # ENGINEERING CALCULATIONS (Vectorized)
            # 1. Add Distance delta for interpolation
            tel['DistanceDelta'] = tel['Distance'].diff().fillna(0)
            
            # 2. Smooth Speed (Savitzky-Golay)
            # PRO FIX: Dynamic window size based on data length to prevent errors on short laps
            window = min(9, len(tel))
            if window > 3:
                from scipy.signal import savgol_filter
                tel['Speed_Smooth'] = savgol_filter(tel['Speed'], window, 2)
            else:
                tel['Speed_Smooth'] = tel['Speed']

            # 3. Derive Accelerations (G-Forces)
            # a = dv/dt
            tel['TimeSec'] = tel['Time'].dt.total_seconds()
            tel['G_Long'] = np.gradient(tel['Speed_Smooth'] / 3.6, tel['TimeSec']) / 9.81
            
            # Store in cache
            result = {"telemetry": tel, "lap_data": lap}
            self.telemetry_cache[cache_key] = result
            return result

        except Exception as e:
            logging.warning(f"⚠️ Telemetry fail for {driver}: {e}")
            return None