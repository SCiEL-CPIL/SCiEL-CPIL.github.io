import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # Build edge set for O(1) lookup
    edge_set = set()
    for e in P['edges']:
        u, v = e[0], e[1]
        edge_set.add((min(u, v), max(u, v)))

    def make_cycle(step):
        cycle = []
        curr = 0
        for _ in range(n):
            cycle.append(int(curr))
            curr = (curr + step) % n
        return cycle

    def validate(cycles):
        if not cycles:
            return False
        
        # Check if all cycles have the same vertex set of size n
        vset = set(cycles[0])
        if len(vset) != n:
            return False
        
        seen_edges = set()
        for c in cycles:
            # Check vertex set and distinctness
            if set(c) != vset:
                return False
            if len(c) != n:
                return False
            if len(set(c)) != n:
                return False
            
            # Check edges
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                key = (min(u, v), max(u, v))
                
                # Check if edge exists in host graph
                if key not in edge_set:
                    return False
                
                # Check if edge is already used
                if key in seen_edges:
                    return False
                seen_edges.add(key)
        
        return True

    # The graph is the circulant C_21(1, 2, 5).
    # It has 63 edges. Each Hamiltonian cycle has 21 edges.
    # Max edge-disjoint Hamiltonian cycles = 63 // 21 = 3.
    # The steps 1, 2, 5 generate three such cycles that partition the edge set.
    
    steps = [1, 2, 5]
    cycles = [make_cycle(s) for s in steps]
    
    # Final validation to ensure correctness
    if validate(cycles):
        return [list(c) for c in cycles]
    else:
        # Fallback: return empty list if validation fails (should not happen)
        return []

# EVOLVE-BLOCK-END
