"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import random
from collections import defaultdict

def construct():
    # The graph is a circulant graph C_21(1, 2, 5).
    # It is 6-regular, so it can be decomposed into 3 edge-disjoint Hamiltonian cycles.
    # Cycle 1: Step 1
    cycle1 = [i for i in range(21)]
    # Cycle 2: Step 2
    cycle2 = []
    v = 0
    for _ in range(21):
        cycle2.append(v)
        v = (v + 2) % 21
    # Cycle 3: Step 5
    cycle3 = []
    v = 0
    for _ in range(21):
        cycle3.append(v)
        v = (v + 5) % 21
    
    cycles = [cycle1, cycle2, cycle3]
    
    # Verify validity (optional but safe)
    edges_set = set()
    for u, v in P['edges']:
        edges_set.add((u, v))
        edges_set.add((v, u))
        
    used_edges = set()
    valid = True
    for c in cycles:
        if len(c) != 21:
            valid = False
            break
        if len(set(c)) != 21:
            valid = False
            break
        for i in range(21):
            u = c[i]
            v = c[(i + 1) % 21]
            if (u, v) not in edges_set:
                valid = False
                break
            if (u, v) in used_edges or (v, u) in used_edges:
                valid = False
                break
            used_edges.add((u, v))
            used_edges.add((v, u))
        if not valid:
            break
            
    if valid:
        return cycles
    else:
        # Fallback to a single known valid cycle if something goes wrong (should not happen)
        return [cycle1]

# EVOLVE-BLOCK-END
