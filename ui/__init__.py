import os
import fastf1

class DataManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        
        # 1. Create the folder if it's missing on the server
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # 2. Now FastF1 can safely enable it
        fastf1.Cache.enable_cache(cache_dir)
        
        self.session = None
        self.laps = None
        self.telemetry_cache = {}
