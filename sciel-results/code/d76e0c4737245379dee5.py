"""Optimal: 3 edge-disjoint Hamiltonian cycles in the 6-regular 21-vertex circulant graph (63 edges / 21 per cycle = 3 max)."""
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Precompute the three circulant Hamiltonian cycles: steps 1, 2, 5
    # These are edge-disjoint and cover all 63 edges (3 * 21 = 63)
    # This is the theoretical maximum since each Hamiltonian cycle uses exactly 21 edges.
    cycles = []
    for k in (1, 2, 5):
        cyc = [0]
        cur = 0
        for _ in range(n - 1):
            cur = (cur + k) % n
            cyc.append(cur)
        cycles.append(cyc)
    
    # Return as list of Python ints in lists
    return [[int(v) for v in cyc] for cyc in cycles]
# EVOLVE-BLOCK-END
