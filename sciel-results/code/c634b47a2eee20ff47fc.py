import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. LUW Graph Construction (Fallback)
    # ... (same as before) ...
    
    # 2. Random Search
    # Try to find a 5-regular graph of girth 9 with n < 6250
    # Start with n = 1116, then increase
    
    best_n = None
    best_edges = None
    
    # We can try a few random graphs
    for n in [1116, 1200, 1500, 2000, 3000]:
        if time.time() > deadline:
            break
        # Generate random 5-regular graph
        # Check girth
        # If girth >= 9, update best
        pass
    
    if best_n is not None:
        return best_n, best_edges
    else:
        return luw_n, luw_edges
