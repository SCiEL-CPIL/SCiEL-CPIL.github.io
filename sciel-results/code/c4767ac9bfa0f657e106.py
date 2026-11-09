"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    # The graph is a circulant graph C_21(1, 2, 5).
    # It is 6-regular with 21 vertices.
    # Max edge-disjoint Hamiltonian cycles = 6/2 = 3.
    # We can construct three cycles using steps 1, 2, and 5.
    
    n = P['n']
    
    # Cycle 1: step 1
    c1 = list(range(n))
    
    # Cycle 2: step 2
    c2 = []
    cur = 0
    for _ in range(n):
        c2.append(cur)
        cur = (cur + 2) % n
    
    # Cycle 3: step 5
    c3 = []
    cur = 0
    for _ in range(n):
        c3.append(cur)
        cur = (cur + 5) % n
    
    # Verify validity (optional but good for safety)
    # Check if all vertices are distinct
    if len(set(c1)) != n or len(set(c2)) != n or len(set(c3)) != n:
        # Fallback to seed if something went wrong (should not happen)
        return [list(range(n))]
        
    # Check if edges exist in P['edges']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
    def is_valid_cycle(cycle):
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if (min(u, v), max(u, v)) not in edge_set:
                return False
        return True

    if not (is_valid_cycle(c1) and is_valid_cycle(c2) and is_valid_cycle(c3)):
        return [list(range(n))]

    # Check edge disjointness
    used_edges = set()
    for c in [c1, c2, c3]:
        for i in range(n):
            u = c[i]
            v = c[(i + 1) % n]
            key = (min(u, v), max(u, v))
            if key in used_edges:
                # Conflict, return what we can
                break
            used_edges.add(key)
        else:
            continue
        break
    else:
        return [c1, c2, c3]
        
    # If conflict found (should not happen for this specific graph), return single cycle
    return [c1]

# EVOLVE-BLOCK-END
