import time
import random
import numpy as np
from collections import defaultdict

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix
    try:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        D = np.sum(diff**2, axis=2).astype(np.int32)
    except MemoryError:
        D = None

    # Maintain for each point in S, a set of distances to other points in S
    # dist_set[s_idx] = set of squared distances from s_idx to other points in S
    dist_sets = {}
    # Also maintain for quick lookup: for each point p (not in S), 
    # we need to check if d(p, s) is in dist_sets[s] for any s in S
    
    def can_add(p, S_list, dist_sets):
        if not S_list:
            return True
            
        # 1. Distances from p to all in S must be distinct
        d_p_S = D[p, S_list]
        if len(np.unique(d_p_S)) != len(S_list):
            return False
        
        # 2. For each s in S, d(p, s) must not be in dist_sets[s]
        for k, s_idx in enumerate(S_list):
            d_val = d_p_S[k]
            if d_val in dist_sets[s_idx]:
                return False
                
        return True

    def add_point(p, S_list, dist_sets):
        # Add p to S_list and update dist_sets
        S_list.append(p)
        dist_sets[p] = set()
        for s_idx in S_list[:-1]:
            d_val = D[p, s_idx]
            dist_sets[p].add(d_val)
            dist_sets[s_idx].add(d_val)

    def remove_point(p, S_list, dist_sets):
        # Remove p from S_list and update dist_sets
        S_list.remove(p)
        for s_idx in S_list:
            d_val = D[p, s_idx]
            dist_sets[s_idx].discard(d_val)
        del dist_sets[p]

    best_S = []
    best_size = 0
    
    # Initial greedy
    S = []
    dist_sets = {}
    candidates = list(range(M))
    random.shuffle(candidates)
    for p in candidates:
        if can_add(p, S, dist_sets):
            add_point(p, S, dist_sets)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    # Keep best state
    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    while time.time() < deadline - 5:
        if random.random() < 0.5 and best_size > 0:
            # Perturb Best
            S = best_S.copy()
            dist_sets = {k: set(v) for k, v in best_dist_sets.items()}
            remove_count = random.randint(1, min(5, len(S)))
            to_remove = random.sample(S, remove_count)
            for r in to_remove:
                remove_point(r, S, dist_sets)
            
            candidates = [p for p in range(M) if p not in S]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(p, S, dist_sets):
                    add_point(p, S, dist_sets)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
        else:
            # Full Restart
            S = []
            dist_sets = {}
            candidates = list(range(M))
            random.shuffle(candidates)
            for p in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(p, S, dist_sets):
                    add_point(p, S, dist_sets)
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    return [tuple(points[i]) for i in best_S]
