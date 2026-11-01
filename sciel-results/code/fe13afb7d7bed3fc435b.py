import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep safety margin
    
    # Generate candidate vectors with squared norm 4
    # Type 1: One entry is 2 or -2
    candidates = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four entries are 1 or -1
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute dot products
    # dot[i, j] = candidates[i] . candidates[j]
    # Conflict if dot[i, j] > 2
    # Note: dot[i, i] = 4, so no self-conflict in terms of "different vectors"
    # but we treat graph as simple.
    
    # To save memory and time, we can compute conflicts on the fly or use bitsets.
    # n=5302. n^2 = ~28M. Fits in memory.
    dots = candidates @ candidates.T
    
    # Conflict matrix: True if dot > 2
    # We want to find Max Independent Set in graph where edges are conflicts.
    # i.e., we want a set S such that for all i, j in S, i != j, dot[i, j] <= 2.
    
    # Greedy MIS heuristic
    # We will run multiple passes with different random orderings
    
    best_set = []
    best_size = 0
    
    # Precompute neighbor lists for speed?
    # Or just use matrix slicing.
    
    # Number of restarts depends on time
    # Estimate time for one run.
    # n=5302.
    # Loop n times. Each step: find min degree or pick random.
    # If we use random order, it's fast.
    
    num_restarts = 0
    while time.time() < deadline - 10:
        # Random permutation of indices
        order = np.random.permutation(n)
        
        available = np.ones(n, dtype=bool)
        current_set = []
        
        for idx in order:
            if not available[idx]:
                continue
                
            # Check if we can add idx
            # We need to ensure it doesn't conflict with any already added
            # Actually, standard greedy:
            # Pick node, remove it and neighbors.
            # This guarantees independence.
            
            current_set.append(idx)
            
            # Remove neighbors
            # neighbors are j where dot[idx, j] > 2
            # We can precompute conflict masks for each node?
            # 5302 * 5302 bits = ~3.5 MB. OK.
            
            # For speed in loop, let's precompute conflict lists
            # But we are in the restart loop. Precompute once outside?
            
            # Let's restructure.
            pass
            
    # Better approach: Precompute conflict lists once.
    # conflicts[i] = list of j such that dot[i, j] > 2
    
    conflicts = []
    for i in range(n):
        # Find j where dot[i, j] > 2
        # Exclude i itself? dot[i, i]=4 > 2.
        # In MIS, we don't add i to its own conflict list.
        mask = dots[i] > 2
        mask[i] = False
        conflicts.append(np.where(mask)[0])
    
    # Now run greedy
    best_set = []
    best_size = 0
    
    while time.time() < deadline - 5:
        order = np.random.permutation(n)
        available = np.ones(n, dtype=bool)
        current_set = []
        
        for idx in order:
            if not available[idx]:
                continue
            
            current_set.append(idx)
            
            # Mark idx and its neighbors as unavailable
            available[idx] = False
            for nei in conflicts[idx]:
                available[nei] = False
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    if best_size > 0:
        return candidates[best_set]
    else:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
