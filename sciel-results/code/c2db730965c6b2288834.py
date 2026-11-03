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
                
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        
        while remaining:
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            best_v = random.choice(candidates)
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
            
        return current_set

    def local_search(iset, steps=5000):
        S = iset.copy()
        current_size = len(S)
        best_size = current_size
        best_set = S.copy()
        
        T = 1.0
        cooling = 0.999
        no_improve = 0
        
        for step in range(steps):
            v = random.randint(0, n_sup - 1)
            
            if v in S:
                if no_improve > 500 and random.random() < 0.1:
                    to_remove = random.sample(list(S), min(3, len(S)))
                    S.difference_update(to_remove)
                    current_size -= len(to_remove)
                    no_improve = 0
                continue
                
            N = adj[v].intersection(S)
            
            if len(N) == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
                    no_improve = 0
            else:
                delta = 1 - len(N)
                if delta > 0:
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                        no_improve = 0
                else:
                    if delta == 0:
                        if random.random() < 0.5:
                            S.difference_update(N)
                            S.add(v)
                    else:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S.difference_update(N)
                            S.add(v)
                            current_size += delta
                            
            if current_size == best_size:
                no_improve += 1
            else:
                no_improve = 0
                
            T *= cooling
            
        return best_set

    best_global = set()
    best_global_size = 0
    
    start_time = time.time()
    time_limit = 110.0
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        init_set = greedy_is()
        improved_set = local_search(init_set, steps=5000)
        
        if len(improved_set) > best_global_size:
            best_global_size = len(improved_set)
            best_global = improved_set.copy()
            
    if time.time() - start_time < time_limit - 5.0:
        final_set = local_search(best_global, steps=10000)
        if len(final_set) > best_global_size:
            best_global = final_set
            best_global_size = len(final_set)

    vectors = []
    for idx in best_global:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
