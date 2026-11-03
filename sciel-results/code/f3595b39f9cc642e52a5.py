import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # 1. Generate all weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    vectors = []
    vec_supports = []
    
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            vec_supports.append(sup)
            
    N = len(vectors)
    
    # 2. Build Conflict Graph
    # Conflict if dot product > 2.
    # For weight-4 vectors, this means dot product == 3 (since 4 implies identical).
    # We build adjacency list of conflicting vector indices.
    
    adj = [set() for _ in range(N)]
    
    # Group vector indices by support
    sup_to_vecs = {}
    for idx, sup in enumerate(vec_supports):
        if sup not in sup_to_vecs:
            sup_to_vecs[sup] = []
        sup_to_vecs[sup].append(idx)
        
    # Iterate over pairs of supports that overlap in 3
    # To avoid O(N^2), we iterate over support pairs.
    # Number of support pairs overlapping in 3 is ~5280.
    
    sup_list = list(sup_to_vecs.keys())
    n_sup = len(sup_list)
    
    # Precompute intersection sizes?
    # Or just check on the fly.
    
    for i in range(n_sup):
        s_i = set(sup_list[i])
        vecs_i = sup_to_vecs[sup_list[i]]
        for j in range(i+1, n_sup):
            s_j = set(sup_list[j])
            inter = s_i.intersection(s_j)
            if len(inter) == 3:
                vecs_j = sup_to_vecs[sup_list[j]]
                # Check all pairs of vectors from these two supports
                # 16 x 16 = 256 checks per support pair.
                # Total ~ 5280 * 256 = 1.3M checks. Fast.
                for v_idx_i in vecs_i:
                    v_i = vectors[v_idx_i]
                    for v_idx_j in vecs_j:
                        v_j = vectors[v_idx_j]
                        # Calculate dot product
                        # Only need to check the 3 intersection indices
                        # and the 4th indices? No, 4th indices are different, so they don't contribute.
                        # Dot product is sum over intersection.
                        dot = 0
                        for k in inter:
                            dot += v_i[k] * v_j[k]
                        if dot > 2: # i.e., dot == 3
                            adj[v_idx_i].add(v_idx_j)
                            adj[v_idx_j].add(v_idx_i)
                            
    # 3. Solve Max Independent Set
    # Greedy Initialization
    def greedy_is():
        remaining = set(range(N))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            # Random tie-breaking to vary results
            # Sort by degree?
            # Just pick min degree
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

    # Run multiple greedy attempts to find a good start
    best_set = set()
    best_size = 0
    for _ in range(5):
        s = greedy_is()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Simulated Annealing
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.9999
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    # Precompute list of all vertices for random selection
    all_vertices = list(range(N))
    
    steps = 0
    while time.time() - start_time < time_limit:
        # Pick random v
        v = random.choice(all_vertices)
        
        if v in S:
            continue
            
        # Find neighbors in S
        N_conf = adj[v].intersection(S)
        
        if len(N_conf) == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - len(N_conf)
            if delta > 0:
                S.difference_update(N_conf)
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                # Accept with probability exp(delta/T)
                # Optimization: if delta is very negative, skip
                if delta < -10 * T:
                    pass
                else:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S.difference_update(N_conf)
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
                            
        T *= cooling
        if T < 0.01:
            # Restart with new greedy or perturbation
            # Perturb: remove 10% of S, add random?
            # Just restart SA with new T
            T = 1.0
            # Randomize S slightly to escape local optima
            # Remove some random elements from S
            to_remove = random.sample(list(S), k=max(1, len(S)//10))
            S.difference_update(to_remove)
            current_size = len(S)
            
    # Return vectors from best_set
    result_vecs = [vectors[i] for i in best_set]
    return np.array(result_vecs, dtype=np.int64)
