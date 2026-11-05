import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try to find low rank decomposition for 3x2x5
    # ... optimization logic ...
    
    # If success, build 3x4x5
    # Else schoolbook
