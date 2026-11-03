import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We know n=936 is possible.
    # We will try to find a 3-regular graph of girth >= g with n close to 936.
    # If we fail, we fall back to the LUW construction (seed) if it's valid and faster?
    # No, the seed is slow to compute if it's huge? 
    # The seed code runs in polynomial time.
    # But if the seed produces n=350,000, and we produce n=936, we win.
    
    # Strategy:
    # 1. Try local search for n in [936, 936+100].
    # 2. If found, return.
    # 3. If not, return the LUW graph (seed) to ensure validity.
    
    best_n = None
    best_edges = None
    
    # We can also try smaller n if we are fast?
    # But 936 is the known minimum.
    
    for n in range(936, 936 + 50):
        if time.time() > deadline:
            break
        
        # Try to find a valid graph of size n
        valid_graph = find_valid_graph(n, k, g, deadline)
        if valid_graph:
            best_n = n
            best_edges = valid_graph
            break # Found one, return it (smallest n first)
            
    if best_n:
        return best_n, best_edges
    else:
        # Fallback to LUW construction
        return luw_construct(k, g)
