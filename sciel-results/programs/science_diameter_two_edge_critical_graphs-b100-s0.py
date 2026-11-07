"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    
    # For n=16, the densest minimally diameter 2 graph is K_{8,8}.
    # This graph has n^2/4 = 64 edges.
    # It is known that for even n, K_{n/2, n/2} is edge-minimal for diameter 2
    # and maximizes the edge count among such graphs.
    
    if n < 2:
        return []
        
    half = n // 2
    # Partition vertices into two sets A and B of size half
    # A = {0, ..., half-1}
    # B = {half, ..., n-1}
    
    edges = []
    for u in range(half):
        for v in range(half, n):
            # Ensure types are Python int
            edges.append([int(u), int(v)])
            
    # Check if we have time (should be instant)
    if time.time() > deadline:
        # Fallback to star graph if somehow we timed out (impossible here)
        return [[int(0), int(v)] for v in range(1, n)]
        
    return edges
# EVOLVE-BLOCK-END
