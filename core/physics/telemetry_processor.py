# core/physics/telemetry_processor.py

import fastf1
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
import logging

# Configure Cache
fastf1.Cache.enable_cache('cache/')

class TelemetryEngine:
    """
    The Core Physics Engine for ApexBrain.
    Handles data ingestion, signal processing, and physics derivation.
    """

    def __init__(self, session_key=None):
        self.session = None
        self.laps = None
        self.logger = logging.getLogger("ApexBrain_Telemetry")

    def load_session(self, year, gp, session_type):
        """
        Loads the session and prepares the data structure.
        """
        try:
            self.session = fastf1.get_session(year, gp, session_type)
            self.session.load()
            self.laps = self.session.laps
            self.logger.info(f"Session Loaded: {year} {gp} - {session_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False

    def get_driver_telemetry(self, driver_code: str, lap_number: int = None):
        """
        Retrieves, smooths, and enriches telemetry for a specific driver.
        If lap_number is None, gets the fastest lap.
        """
        try:
            driver_laps = self.laps.pick_driver(driver_code)
            
            if lap_number:
                lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
            else:
                lap = driver_laps.pick_fastest()

            # Get Telemetry with distance interpolation
            tel = lap.get_telemetry()
            
            # 1. ADD DISTANCE/TIME DELTAS
            tel['TimeSec'] = tel['Time'].dt.total_seconds()
            tel['dt'] = tel['TimeSec'].diff().fillna(0.1) # Avoid div/0
            
            # 2. PHYSICS ENRICHMENT
            tel = self._calculate_physics(tel)
            
            # 3. CONVEX HULL (FRICTION CIRCLE)
            hull_area, hull_points = self._calculate_friction_envelope(tel)
            
            return {
                "telemetry": tel,
                "lap_data": lap,
                "hull_area": hull_area,
                "hull_points": hull_points
            }
            
        except IndexError:
            self.logger.warning(f"No data found for driver {driver_code}")
            return None

    def _calculate_physics(self, tel: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to derive G-forces and smooth signals.
        """
        # CONSTANTS
        GRAVITY = 9.81
        
        # A. SMOOTHING (Savitzky-Golay filter)
        # Window length 9, polyorder 2 is standard for 10Hz telemetry
        try:
            tel['Speed_Smooth'] = savgol_filter(tel['Speed'], 9, 2)
        except:
            tel['Speed_Smooth'] = tel['Speed'] # Fallback for short arrays

        # B. LONGITUDINAL G (Acceleration/Braking)
        # Convert km/h to m/s
        v_ms = tel['Speed_Smooth'] / 3.6
        
        # Calculate derivative dv/dt
        acc_ms2 = np.gradient(v_ms, tel['TimeSec'])
        tel['G_Long'] = acc_ms2 / GRAVITY

        # C. LATERAL G (Cornering)
        # If 'LinearAcceleration' (sensor) is missing, estimate via GPS curvature
        # Note: FastF1 creates 'Distance' automatically.
        
        # Calculate curvature (k) = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        # First, we need X and Y coordinates (FastF1 provides X, Y, Z usually)
        if 'X' in tel.columns and 'Y' in tel.columns:
            x = tel['X'].values
            y = tel['Y'].values
            
            # First derivatives
            dx = np.gradient(x)
            dy = np.gradient(y)
            
            # Second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # Curvature
            curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
            
            # Centripetal Acceleration a = v^2 * k
            # Result is signed (left/right) based on curvature sign
            lat_acc_ms2 = (v_ms ** 2) * curvature
            
            # Convert to G and apply a simple smoothing to remove GPS jitters
            tel['G_Lat'] = lat_acc_ms2 / GRAVITY
            tel['G_Lat'] = tel['G_Lat'].rolling(window=5, center=True).mean().fillna(0)
            
            # Clamp unrealistic values (GPS glitches)
            tel['G_Lat'] = tel['G_Lat'].clip(-6.0, 6.0)
            
        else:
            tel['G_Lat'] = 0.0 # Fallback

        return tel

    def _calculate_friction_envelope(self, tel: pd.DataFrame):
        """
        Calculates the Convex Hull Area of the G-G diagram.
        Higher area = Driver is using more of the car's grip availability.
        """
        # Filter low speeds (pit lane, safety car) to isolate racing performance
        racing_points = tel[tel['Speed'] > 80][['G_Lat', 'G_Long']].dropna()
        
        if len(racing_points) < 3:
            return 0.0, None

        try:
            points = racing_points.values
            hull = ConvexHull(points)
            return hull.volume, points[hull.vertices] # hull.volume is Area in 2D
        except Exception as e:
            self.logger.warning(f"Convex Hull calculation failed: {e}")
            return 0.0, None

    def compare_drivers(self, driver_1, driver_2):
        """
        Aligns two drivers' telemetry on Distance for delta analysis.
        """
        d1 = self.get_driver_telemetry(driver_1)
        d2 = self.get_driver_telemetry(driver_2)
        
        if not d1 or not d2:
            return None

        # Create a common distance axis
        t1 = d1['telemetry']
        t2 = d2['telemetry']
        
        # Calculate Delta Time
        # We interpolate Driver 2's time onto Driver 1's distance
        t2_interp_time = np.interp(t1['Distance'], t2['Distance'], t2['TimeSec'])
        delta = t2_interp_time - t1['TimeSec']
        
        return {
            "driver_1": d1,
            "driver_2": d2,
            "delta_time": delta, # Positive means D1 is faster (D2 arrives later)
            "distance_axis": t1['Distance']
        }

if __name__ == "__main__":
    # Quick Test
    engine = TelemetryEngine()
    if engine.load_session(2024, 'Bahrain', 'Q'):
        data = engine.get_driver_telemetry('VER')
        print(f"Max G-Lat: {data['telemetry']['G_Lat'].max():.2f}")
        print(f"Friction Circle Area: {data['hull_area']:.2f}")