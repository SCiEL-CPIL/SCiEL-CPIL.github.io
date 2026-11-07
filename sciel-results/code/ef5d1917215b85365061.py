"""Optimized search for maximum distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Helper for popcount
    try:
        (0).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def evaluate(vertices):
        mask = 0
        for v in vertices:
            mask |= (1 << v)
        degs = set()
        for v in vertices:
            degs.add(popcount(adj[v] & mask))
        return len(degs)

    best_count = -1
    best_vertices = []

    # Strategy: 
    # 1. Quick exhaustive check for large sizes (17 down to 13)
    # 2. Simulated Annealing for all sizes (17 down to 8) with multiple restarts
    
    # 1. Exhaustive for large sizes
    for size in range(n, 12, -1):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(n), size):
            c = evaluate(combo)
            if c > best_count:
                best_count = c
                best_vertices = list(combo)
                if best_count == size:
                    return best_vertices

    # 2. Simulated Annealing
    # We will run SA for each size from 17 down to minv
    # If best_count is already high, we can skip sizes <= best_count
    
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        if best_count >= size:
            continue

        # Run SA for this size
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            current = set(random.sample(range(n), size))
            current_mask = 0
            for v in current:
                current_mask |= (1 << v)
            
            current_score = evaluate(list(current))
            
            if current_score > best_count:
                best_count = current_score
                best_vertices = list(current)
                if best_count == size:
                    return best_vertices

            temp = 2.0
            cooling = 0.99
            
            # SA loop
            while temp > 0.1 and time.time() < deadline:
                # Pick a vertex in and a vertex out
                in_v = random.choice(list(current))
                out_v = random.choice([v for v in range(n) if v not in current])
                
                # Swap
                new_set = set(current)
                new_set.remove(in_v)
                new_set.add(out_v)
                
                new_score = evaluate(list(new_set))
                
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.71828, delta / temp):
                    current = new_set
                    current_score = new_score
                    
                    if current_score > best_count:
                        best_count = current_score
                        best_vertices = list(current)
                        if best_count == size:
                            return best_vertices

                temp *= cooling

    return best_vertices
# EVOLVE-BLOCK-END
