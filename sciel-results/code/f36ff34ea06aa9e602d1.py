import time
import numpy as np
from itertools import combinations, product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all weight-4 vectors
    all_vecs = []
    for pos in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(pos, signs):
                v[p] = s
            all_vecs.append(v)
            
    V = np.array(all_vecs, dtype=np.int64)
    N = len(V)
    
    # 2. Build Conflict Matrix
    # Conflict if dot product is 3 (sharing 3 coords with same signs)
    # Note: Distinct vectors cannot have dot product 4.
    G = V @ V.T
    conflict = (G == 3)
    np.fill_diagonal(conflict, False)
    
    # 3. Solve Max Independent Set
    # Heuristic: Greedy with Random Restarts + Local Search (1-swap)
    # Using vectorized operations for speed.
    
    best_size = 0
    best_set = np.zeros(N, dtype=bool)
    
    # Helper: Evaluate a set (mask)
    # We want to maximize sum(mask)
    
    def get_size(mask):
        return np.sum(mask)
        
    def local_search(mask):
        # Try to add any vertex that doesn't conflict with current set
        # Then try 1-swap: remove one, add one if net gain > 0
        # For simplicity and speed, just do "add if possible" repeatedly?
        # No, "add if possible" leads to maximal, not necessarily maximum.
        # 1-swap is better.
        
        S = mask.copy()
        improved = True
        while improved:
            improved = False
            # Calculate gain for each vertex not in S
            # If we add u, we must remove all neighbors of u in S.
            # Gain = 1 - (number of neighbors of u in S)
            # Neighbors of u in S are S & conflict[u]
            
            # We can vectorize this check for all u not in S
            # neighbors_in_S_count = (S[:, None] & conflict).sum(axis=1) 
            # This creates an NxN intermediate, which is 5280x5280 ~ 27M bools. 
            # This is fast in NumPy.
            
            # Let's compute for all vertices
            # conflict is NxN bool
            # S is Nx1 bool (or N,)
            # neighbor_counts = conflict @ S  (since S is 0/1)
            # But S is bool. Cast to int or use dot.
            S_int = S.astype(np.int8)
            neighbor_counts = conflict @ S_int
            
            # For u in S, we don't care.
            # For u not in S, gain = 1 - neighbor_counts[u]
            # We want gain > 0, i.e., neighbor_counts[u] == 0
            # This is just adding isolated vertices in the remaining graph.
            
            # To do 1-swap:
            # Remove v in S, add u not in S.
            # New size = |S| - 1 + 1 = |S| (no change in size, just swap)
            # We want to increase size.
            # So we look for u not in S such that we can add it by removing k neighbors, 
            # and k < 1? Impossible.
            # So simple 1-swap doesn't increase size.
            
            # We need to look for "Add u, remove neighbors".
            # Size change = 1 - count(neighbors(u) in S).
            # If count is 0, size increases by 1.
            # If count is 1, size stays same.
            # If count > 1, size decreases.
            
            # So, if there is any u not in S with neighbor_counts[u] == 0, we add it.
            # This is just finding isolated vertices in the complement of S?
            # No, isolated in the subgraph induced by V\S? No.
            # u is non-adjacent to all v in S.
            
            candidates = ~S
            # For each candidate u, check if it conflicts with any v in S.
            # conflict[u, :] & S -> any?
            
            # Vectorized check:
            # Does u conflict with any node in S?
            # (conflict @ S_int) > 0
            # If 0, it's a candidate.
            
            has_conflict = (neighbor_counts > 0)
            add_candidates = candidates & ~has_conflict
            
            if np.any(add_candidates):
                # Add all of them? 
                # If we add u1 and u2, they might conflict with each other.
                # So we should add one by one or a maximal independent set of candidates.
                # Since candidates are non-adjacent to S, but may be adjacent to each other.
                # We can just pick the first one to be safe, or use a greedy on candidates.
                # Given time, picking one is a valid step.
                # Or, since candidates are "safe" w.r.t S, we can add a maximal independent set of the subgraph induced by candidates.
                
                # Let's just add one to ensure progress and simplicity.
                idx = np.argmax(add_candidates)
                S[idx] = True
                improved = True
                # Update S_int? No, we recompute in next loop.
                
        return S

    # Greedy Initialization
    # Sort by degree (number of conflicts)
    degrees = conflict.sum(axis=1)
    
    # Try multiple random restarts
    for i in range(100):
        if time.time() > deadline:
            break
            
        if i == 0:
            # Greedy by degree
            order = np.argsort(degrees)
        else:
            # Random
            order = np.random.permutation(N)
            
        S = np.zeros(N, dtype=bool)
        
        # Greedy pass
        for u in order:
            if S[u]:
                continue
            # Check if u conflicts with any in S
            # (conflict[u, :] & S).any()
            # Vectorized: conflict[u] @ S_int?
            # We can maintain S_int
            # But updating S_int inside loop is slow.
            # Alternative: Keep S as set of indices?
            # For N=5280, Python set is okay if we don't do it too many times.
            # But we want NumPy speed.
            
            # Let's stick to the previous logic:
            # We can check `np.any(conflict[u, S])`
            # This is a dot product of a row and a mask.
            if not np.any(conflict[u, S]):
                S[u] = True
                
        # Local Search
        S = local_search(S)
        
        if np.sum(S) > best_size:
            best_size = np.sum(S)
            best_set = S.copy()
            
    # Extract vectors
    indices = np.where(best_set)[0]
    result = V[indices]
    
    return result
