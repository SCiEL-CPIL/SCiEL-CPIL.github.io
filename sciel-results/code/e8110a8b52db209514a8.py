"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Theoretical maximum for triangle-free graph on 16 vertices is K(8,8) with 64 edges.
    # Any triangle would make an edge non-essential (distance remains 2).
    # Thus, we must use K(8,8) to maximize edge count while satisfying essential edge condition.
    
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    n = 16
    edges = []
    
    # Generate all edges between A (0-7) and B (8-15)
    # Using list comprehension for efficiency
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
            
    # Check time, though this is O(64) so it's instantaneous
    if time.time() > deadline:
        # Fallback to a smaller valid graph if somehow too slow (impossible here)
        return [[0, i] for i in range(1, 8)]
        
    return edges
# EVOLVE-BLOCK-END
