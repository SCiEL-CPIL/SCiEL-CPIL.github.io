import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # 1. Greedy Initialization
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0: break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # 2. Quantum-Inspired Simulated Annealing
    S = best_set.copy()
    current_size = len(S)
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 1.0
    cooling = 0.9999
    all_nodes = list(range(n_sup))
    
    while time.time() - start_time < time_limit:
        # Multi-flip move selection
        r = random.random()
        
        if r < 0.8:
            # Single node move
            v = random.choice(all_nodes)
            if v in S:
                continue
            N = adj[v].intersection(S)
            if len(N) == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - len(N)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
        else:
            # Double node move (2-swap)
            if len(S) >= 2:
                u1, u2 = random.sample(list(S), 2)
                S_temp = S - {u1, u2}
                # Try to find v1, v2
                for _ in range(5):
                    v1 = random.choice(all_nodes)
                    v2 = random.choice(all_nodes)
                    if v1 in S or v2 in S or v1 == v2:
                        continue
                    if v2 in adj[v1]:
                        continue
                    if adj[v1].intersection(S_temp) or adj[v2].intersection(S_temp):
                        continue
                    S = S_temp | {v1, v2}
                    current_size = len(S)
                    break
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Construct vectors from best_set
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
