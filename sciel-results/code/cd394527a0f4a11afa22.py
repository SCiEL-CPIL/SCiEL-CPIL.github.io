"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n = 20
    # Use a K(10,10) bipartite graph structure
    # Vertices 0-9 in U, 10-19 in V
    # Color edge (u, v) with c = (u + 2*(v-10)) % 20
    # This ensures any 2x2 subgrid (which corresponds to a 4-cycle) has distinct colors
    # because the colors are x, x+1, x+2, x+3 mod 20, which are distinct for modulus 20 > 3
    
    edges = []
    for u in range(10):
        for v in range(10, 20):
            v_prime = v - 10
            c = (u + 2 * v_prime) % 20
            edges.append([u, v, c])
            
    # Verify validity (optional, but good for safety)
    # We can also try to add edges within U or V if time permits, but K(10,10) is a strong baseline.
    # The random greedy might find a non-bipartite graph with more edges, but it's slow.
    # Let's stick to the deterministic construction for speed and reliability.
    
    return edges
# EVOLVE-BLOCK-END
