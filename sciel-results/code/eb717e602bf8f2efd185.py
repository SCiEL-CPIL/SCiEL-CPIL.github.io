import sys
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(tuple(sorted((u, v))))
    
    # This graph is the circulant C_21(1,2,5).
    # Construct 3 edge-disjoint Hamiltonian cycles using step sizes 1, 2, 5.
    # Step s cycle: 0, s, 2s, 3s, ..., 20s (mod 21)
    # Edges in step-s cycle: {(i, (i+s) mod 21) for i in 0..20}
    # Steps 1, 2, 5 are pairwise edge-disjoint since 1, 2, 5 mod 21 are distinct
    # and none equals 21-s' for any pair (no edge collision).
    
    cycles = []
    used_edges = set()
    
    for step in [1, 2, 5]:
        cycle = [(i * step) % n for i in range(n)]
        # Verify all edges are valid and not previously used
        valid = True
        for j in range(n):
            u = cycle[j]
            v = cycle[(j + 1) % n]
            e = (min(u, v), max(u, v))
            if e not in edge_set or e in used_edges:
                valid = False
                break
        if valid:
            for j in range(n):
                u = cycle[j]
                v = cycle[(j + 1) % n]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append([int(v) for v in cycle])
    
    return cycles
# EVOLVE-BLOCK-END
