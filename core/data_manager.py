import fastf1
import pandas as pd
import numpy as np
import logging
import streamlit as st
import os  # Required for directory management
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
        
        # FIX: Programmatically create the cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"📁 Created missing cache directory at: {cache_dir}")
            
        # Now enable the cache safely
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
            d_laps = self.laps.pick_driver(driver)
            
            if lap_number:
                lap = d_laps[d_laps['LapNumber'] == lap_number].iloc[0]
            else:
                lap = d_laps.pick_fastest()

            tel = lap.get_telemetry()
            
            tel['DistanceDelta'] = tel['Distance'].diff().fillna(0)
            
            window = min(9, len(tel))
            if window > 3:
                from scipy.signal import savgol_filter
                tel['Speed_Smooth'] = savgol_filter(tel['Speed'], window, 2)
            else:
                tel['Speed_Smooth'] = tel['Speed']

            tel['TimeSec'] = tel['Time'].dt.total_seconds()
            tel['G_Long'] = np.gradient(tel['Speed_Smooth'] / 3.6, tel['TimeSec']) / 9.81
            
            result = {"telemetry": tel, "lap_data": lap}
            self.telemetry_cache[cache_key] = result
            return result

        except Exception as e:
            logging.warning(f"⚠️ Telemetry fail for {driver}: {e}")
            return None
