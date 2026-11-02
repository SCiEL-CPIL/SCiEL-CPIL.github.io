import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict graph: edge if M[i,j] > 2 (incompatible)
    # We want maximum independent set: no two selected have M[i,j] > 2
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Compatibility: 1 if compatible (M[i,j] <= 2)
    # Note: M[i,i] = 4 or 16, so diagonal is conflict with itself, but we handle selection carefully.
    # For i != j, compatible if M[i,j] <= 2.
    compat = (M <= 2).astype(np.int8)
    np.fill_diagonal(compat, 0) # Self is not compatible for pairing purposes in this logic, but we select unique indices.
    
    # Greedy initialization with multiple orderings
    rng = np.random.default_rng(12345)
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multiple Random Greedy Restarts
    # We will run many random permutations and greedily select.
    # To speed up, we can precompute adjacency lists or use vectorized checks if possible, 
    # but for 5302 nodes, O(N^2) per restart is ~28M ops. 100 restarts = 2.8B ops. Too slow in Python loops.
    # We need a faster greedy or a different approach.
    
    # Let's try a smarter greedy: 
    # 1. Sort by degree (compat count) descending.
    # 2. Pick highest degree, remove neighbors.
    # This is standard "Degree" greedy for MIS.
    
    # Precompute degrees
    degrees = compat.sum(axis=1)
    
    # Orderings: 
    # 1. Static degree order
    # 2. Random permutations (limited)
    
    orderings = [np.argsort(-degrees)]
    
    # Add some random orderings, but limit the number to keep time safe
    # Let's try 50 random orderings
    for _ in range(50):
        orderings.append(rng.permutation(N))
    
    for oi, order in enumerate(orderings):
        if time.time() > deadline - 20:
            break
            
        selected = []
        selected_mask = np.zeros(N, dtype=bool)
        
        # Greedy selection
        # Optimization: Instead of checking all selected for each candidate, 
        # we can maintain a "forbidden" mask.
        # forbidden[i] = 1 if i is incompatible with ANY selected vertex.
        # Initially 0.
        # When we select v, we OR the conflict row of v into forbidden.
        # We pick the next v from `order` such that forbidden[v] == 0.
        
        forbidden = np.zeros(N, dtype=np.int8)
        
        for idx in order:
            if forbidden[idx]:
                continue
                
            # Select idx
            selected.append(idx)
            selected_mask[idx] = True
            
            # Update forbidden: any neighbor of idx becomes forbidden
            # conflict[idx] is 1 for incompatible nodes.
            # We want to mark incompatible nodes as forbidden.
            # forbidden |= conflict[idx]
            # Since conflict is int8, this is a vectorized OR.
            forbidden |= conflict[idx]
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    # Local Search Improvement
    # Try to swap elements to increase size? 
    # Or try to add elements that were missed?
    # The greedy above is "first fit" in the given order. 
    # A "best fit" or "max min" might be better but harder to vectorize simply.
    
    # Let's try a second phase: Local Search on the best set found.
    # We want to see if we can replace a subset of selected with more unselected.
    # This is complex. 
    # Alternative: Run the greedy again but starting from the best_set? 
    # Or, use the best_set to initialize a "conflict" state and try to add more? 
    # The greedy already adds all possible in that order. 
    
    # Let's try a different heuristic: 
    # "Max-Min" or "Least Constraints"
    # At each step, pick the vertex that has the most compatible unselected neighbors?
    # This is harder to implement efficiently in a loop.
    
    # Given the time constraints, the vectorized greedy with random restarts is a solid baseline.
    # Let's increase the number of restarts if we have time.
    
    # Check time
    if time.time() < deadline - 10:
        # Run more random restarts
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            order = rng.permutation(N)
            selected = []
            forbidden = np.zeros(N, dtype=np.int8)
            
            for idx in order:
                if forbidden[idx]:
                    continue
                selected.append(idx)
                forbidden |= conflict[idx]
                
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
                
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[best_set].astype(np.int64)
    
    # Verify validity
    n = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    # Optimized verification
    if n > 1:
        # Compute all pairwise distances squared
        # diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # np.fill_diagonal(dist_sq, np.inf)
        # min_dist = np.min(dist_sq)
        # if min_dist < max_norm_sq: valid = False
        
        # Simpler loop for verification since n is likely < 100
        for i in range(n):
            for j in range(i + 1, n):
                diff = result[i] - result[j]
                if np.dot(diff, diff) < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return result
