import os
import fastf1

class DataManager:
    def __init__(self):
        # Define the relative path to your cache folder
        cache_dir = 'cache' 
        
        # FIX: Create the directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # Now enable the cache safely
        fastf1.Cache.enable_cache(cache_dir)
