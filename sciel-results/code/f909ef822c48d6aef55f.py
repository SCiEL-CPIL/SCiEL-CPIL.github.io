import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Generate Candidate Vectors
    # Weight 2 vectors
    cand_w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                cand_w2.append(tuple(v))
    
    # Weight 3 vectors
    cand_w3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    cand_w3.append(tuple(v))
    
    all_cands = cand_w2 + cand_w3
    n_cands = len(all_cands)
    
    # 2. Define Conflict for R = sqrt(3)
    # Conflict if dist^2 < 3 (i.e., 1 or 2)
    # We check if a set S is valid for R=sqrt(3)
    
    # Precompute pairwise squared distances for all candidates?
    # n_cands is ~1540. 1540^2 is ~2.3M. This is fine.
    # But we only need conflicts.
    
    # Build conflict bitsets for faster independent set search
    # conflicts[i] is a bitset of nodes conflicting with i
    
    # To save time, we can just run a heuristic search directly.
    # Greedy + Local Search
    
    best_set = set()
    best_size = 0
    
    # Fallback: The seed solution (all weight 2 vectors)
    # This is valid for R=sqrt(2).
    # Size is 220.
    seed_set = set(cand_w2)
    seed_size = len(seed_set)
    
    # If we find a set for R=sqrt(3) with size > 220, we use it.
    # Note: A set valid for R=sqrt(3) is also valid for R=sqrt(2)?
    # No. If min dist >= sqrt(3), it is >= sqrt(2).
    # But the condition is min dist >= MAX NORM.
    # If the set contains weight 3 vectors, max norm is sqrt(3).
    # So we must check min dist >= sqrt(3).
    # If the set contains ONLY weight 2 vectors, max norm is sqrt(2).
    # We must check min dist >= sqrt(2).
    
    # The seed set has min dist sqrt(2). It is valid.
    
    # Let's try to find a larger set.
    # We will search for an independent set in the conflict graph defined by dist^2 < 3.
    # If we find a set S of size K, and S contains any weight 3 vector, it is valid if min dist >= sqrt(3).
    # If S contains only weight 2 vectors, it is valid if min dist >= sqrt(2).
    # But if we use the conflict graph for dist^2 < 3, then any independent set in it has min dist >= sqrt(3).
    # Such a set is valid regardless of composition (since sqrt(3) > sqrt(2)).
    
    # So we just need to find the Maximum Independent Set (MIS) in the graph where edges connect vectors with dist^2 < 3.
    # If MIS size > 220, we return it.
    # Else we return the seed.
    
    # Build adjacency list / bitsets
    # For speed, we'll use a simple greedy with random restarts.
    
    # Precompute conflict pairs
    conflicts = [set() for _ in range(n_cands)]
    
    # To speed up, we can check distances.
    # dist^2 < 3 means dist^2 is 1 or 2.
    # This happens if:
    # 1. They are identical (dist 0) - not possible in set of unique tuples.
    # 2. They differ in 1 coordinate by 1?
    #    e.g., (1,1,0) and (1,0,0)? No, weights differ.
    #    (1,1,1) and (1,1,0)? No, weight 3 vs 2.
    #    (1,1,1) and (1,1,-1)? Dist^2 = 4.
    #    (1,1,1) and (1,1,2)? Not integer candidate.
    
    # Let's just compute dist^2 for all pairs.
    # 1540 * 1540 / 2 = ~1.2M pairs.
    # NumPy vectorization can help.
    
    arr = np.array(all_cands)
    # Compute squared distances
    # We can do this in chunks to avoid memory blowup, but 1540x1540x11 is small.
    # 1540*1540*8 bytes = ~19MB. Fine.
    
    # dist_sq[i, j] = sum((arr[i]-arr[j])**2)
    # We want pairs where dist_sq < 3.
    
    # To save time, we can filter candidates first?
    # No, just compute.
    
    # Vectorized distance calculation
    # This might take a few seconds.
    # If time is tight, we can skip the search and return seed.
    
    if time.time() > deadline - 5.0:
        return np.array(list(seed_set), dtype=np.int64)
    
    # Compute conflict matrix
    # dist_sq = np.sum((arr[:, np.newaxis, :] - arr[np.newaxis, :, :])**2, axis=2)
    # This creates a 1540x1540x11 array. 1540*1540*11*8 = ~210MB.
    # This might be too slow or memory heavy.
    
    # Alternative: Loop over i, compute dist to all j > i.
    # 1540 iterations. Each 1540*11 ops. ~25k ops. Total 38M ops. Fast in Python? No.
    # Use NumPy per row.
    
    conflicts_list = [set() for _ in range(n_cands)]
    
    for i in range(n_cands):
        if time.time() > deadline - 10.0:
            break
        # Compute dist from i to all j
        diffs = arr[i] - arr
        dists_sq = np.sum(diffs**2, axis=1)
        # Find j where dists_sq < 3 and j != i
        mask = (dists_sq < 3)
        mask[i] = False
        conflicting_indices = np.where(mask)[0]
        for j in conflicting_indices:
            conflicts_list[i].add(j)
            conflicts_list[j].add(i)
            
    if time.time() > deadline - 10.0:
        return np.array(list(seed_set), dtype=np.int64)
    
    # Now we have the conflict graph.
    # Find a large independent set.
    # Greedy heuristic:
    # 1. Start with empty set.
    # 2. Repeatedly pick a node with minimum degree in the remaining graph?
    #    Or random node, add to set, remove it and its neighbors.
    # 3. Local search: Try to swap nodes.
    
    def greedy_is(conflicts, n):
        remaining = set(range(n))
        selected = set()
        while remaining:
            # Pick node with min degree in remaining
            min_deg = float('inf')
            best_node = None
            # Randomize to allow different runs?
            # For speed, just pick first min.
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
            selected.add(best_node)
            # Remove best_node and its neighbors
            to_remove = set(conflicts[best_node])
            to_remove.add(best_node)
            remaining -= to_remove
        return selected
        
    best_size = 0
    best_set = set()
    
    # Run multiple random restarts
    # To make it random, we can shuffle the order of nodes or break ties randomly.
    # Simpler: Randomly permute nodes, then greedy.
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 5.0:
            break
        
        # Shuffle indices
        indices = list(range(n_cands))
        np.random.shuffle(indices)
        
        # Greedy with shuffled order
        remaining = set(indices)
        selected = set()
        while remaining:
            # Pick first available in shuffled order? No, that's just greedy.
            # Better: Pick node with min degree among remaining.
            # To add randomness, we can pick randomly among the k-min degree nodes.
            
            # For speed, just pick min degree.
            min_deg = float('inf')
            candidates = []
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [node]
                elif deg == min_deg:
                    candidates.append(node)
            
            # Pick random from candidates
            best_node = candidates[np.random.randint(len(candidates))]
            selected.add(best_node)
            to_remove = set(conflicts[best_node])
            to_remove.add(best_node)
            remaining -= to_remove
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
        # Check if we already beat the seed
        if best_size > seed_size:
            # We found a better solution.
            # We can try to polish it, but if time is tight, return it.
            break
            
    if best_size > seed_size:
        # Return the vectors corresponding to best_set
        result = [all_cands[i] for i in best_set]
        return np.array(result, dtype=np.int64)
    else:
        # Return seed
        return np.array(list(seed_set), dtype=np.int64)

# EVOLVE-BLOCK-END
