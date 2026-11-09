import time
import numpy as np
from itertools import combinations, product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_remaining():
        return deadline - time.time()

    # Candidate pool: All non-zero integer vectors in R^11 with norm^2 = 4.
    # This allows for a much larger potential kissing configuration than norm^2=2 or 1.
    # Condition: min pairwise distance >= max norm.
    # If we select a set S of vectors all with norm^2 = 4, then max norm = 2.
    # We need min_{u,v in S, u!=v} ||u-v|| >= 2.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 8 - 2<u,v>.
    # Condition: 8 - 2<u,v> >= 4  =>  4 >= 2<u,v>  =>  <u,v> <= 2.
    # So we need to find the maximum size subset of norm^2=4 integer vectors such that
    # the inner product of any two distinct vectors is at most 2.
    
    # Types of norm^2=4 vectors:
    # 1. Type A: One coordinate is ±2, rest 0. (22 vectors)
    # 2. Type B: Four coordinates are ±1, rest 0. (C(11,4) * 2^4 = 330 * 16 = 5280 vectors)
    
    # Let's generate all candidates.
    candidates = []
    
    # Type A
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot products
    # 5302 x 5302 matrix. 28M entries. Numpy handles this fast.
    dots = candidates @ candidates.T
    
    # Conflict if dot > 2.
    # Note: Diagonal is 4. We ignore self.
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # We want the Maximum Independent Set (MIS) of this graph.
    # Greedy heuristic:
    # 1. Compute degrees (number of conflicts).
    # 2. Sort nodes by degree (ascending).
    # 3. Iterate through sorted nodes. If a node is not "blocked" (i.e., not conflicting with an already selected node), select it and block its neighbors.
    
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(degrees)
    
    blocked = np.zeros(n_cand, dtype=bool)
    selected_indices = []
    
    # Optimization: 
    # Instead of checking `if not blocked[i]` which is O(1), and then blocking neighbors O(degree),
    # we can just do it. 5302 nodes is small.
    
    for i in order:
        if blocked[i]:
            continue
        selected_indices.append(i)
        # Block all neighbors
        neighbors = np.where(conflict[i])[0]
        blocked[neighbors] = True
        blocked[i] = True
        
    # Check time
    if time_remaining() < 10.0:
         # If we ran out of time, return what we have (might be partial, but let's validate)
         pass
         
    selected_vectors = candidates[selected_indices]
    
    # Validation
    if len(selected_vectors) == 0:
        return np.array([], dtype=np.int64)
        
    n = len(selected_vectors)
    # We know all norms are 4.
    # Check min distance.
    # For n ~ 1000+, O(n^2) distance check is 1M ops. Fast.
    # But let's be safe.
    # Actually, the greedy ensures no conflicts in the graph defined by dot > 2.
    # So min distance is guaranteed >= 2 IF the graph construction was correct.
    # Let's do a spot check or full check if n is small enough, otherwise trust the graph.
    # 5302^2 is 28M. Full check is fine.
    
    # However, we must ensure we didn't miss any constraints.
    # The constraint was dot <= 2.
    # If we selected nodes such that no pair has dot > 2, then for any pair, dot <= 2.
    # Then dist^2 = 8 - 2*dot >= 8 - 2(2) = 4.
    # Dist >= 2.
    # Max norm is 2.
    # So condition holds.
    
    # One edge case: Did we include duplicates? No, generated uniquely.
    # Did we include zero vector? No, norm^2=4.
    
    return selected_vectors
