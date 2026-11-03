import itertools
import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_vectors = []
    best_score = 0
    
    # Weights to try
    weights_to_try = [3, 4]
    
    for w in weights_to_try:
        if time.time() > deadline - 30:
            break
            
        # 1. Generate candidates
        candidates = []
        for support in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, s in zip(support, signs):
                    v[idx] = s
                candidates.append(v)
        
        n_cands = len(candidates)
        if n_cands == 0:
            continue
        
        # 2. Precompute conflict graph
        # conflicts[i] contains indices j such that candidates[i] and candidates[j] conflict
        # Conflict condition: distance squared < w
        # dist^2 = 4 * (mismatches in intersection) + 2 * (w - |intersection|)
        # We need dist^2 >= w
        # So conflict if 4 * m + 2 * (w - k) < w  => 4m + 2w - 2k < w => 4m + w < 2k => 2m + w/2 < k
        # Where k is intersection size, m is mismatches in intersection.
        
        # To speed up, we can represent vectors by their support and signs.
        # But direct check is fine for N < 6000.
        
        conflicts = [set() for _ in range(n_cands)]
        
        # Optimization: Group by support?
        # Vectors with disjoint supports never conflict (dist^2 = 2w >= w).
        # Vectors with same support:
        #   k = w. Conflict if 2m + w/2 < w => 2m < w/2 => m < w/4.
        #   For w=3, m < 0.75 => m=0. So if all signs match (m=0), conflict?
        #   Wait, if m=0, they are identical. Distance 0. Conflict.
        #   If m=1, 2(1) + 1.5 = 3.5 < 3? No.
        #   Let's re-verify w=3, same support.
        #   k=3. 2m + 1.5 < 3 => 2m < 1.5 => m=0.
        #   So if m=0 (identical), conflict.
        #   If m=1 (one sign diff), 2(1)+1.5 = 3.5 >= 3. No conflict.
        #   So distinct vectors with same support are OK for w=3.
        
        # For w=4, same support.
        # k=4. 2m + 2 < 4 => 2m < 2 => m=0.
        # So identical is conflict. m=1 => 2+2=4 >= 4. OK.
        
        # General loop
        # To make it faster, we can iterate i < j.
        # But Python loops are slow.
        # N=5280. N^2/2 = 14M pairs.
        # Each pair check:
        #   Find intersection.
        #   This is expensive if done naively.
        
        # Better representation:
        # For each vector, store support as a frozenset or tuple, and signs.
        # Or, since D=11 is small, use bitmasks for support.
        
        support_masks = []
        sign_masks = [] # Or just the vector itself
        
        for v in candidates:
            mask = 0
            for i in range(D):
                if v[i] != 0:
                    mask |= (1 << i)
            support_masks.append(mask)
            # Store signs in a tuple or similar for quick access?
            # Actually, we can just use the vector for the distance check if we optimize it.
            
        # Let's try a different approach for conflict checking:
        # Pre-compute for each pair of indices? No.
        
        # Given the time limit, I will use a simpler heuristic for w=4 if it's too slow.
        # But 14M checks in Python might take ~10-20s.
        # Let's optimize the inner loop.
        
        # We can pre-compute "signature" for each vector to speed up intersection?
        # Or just use numpy?
        # N x D matrix.
        # Distances between all pairs:
        # (X @ X.T) gives dot products.
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = w + w - 2 u.v = 2w - 2 u.v.
        # Conflict if 2w - 2 u.v < w  =>  w < 2 u.v  =>  u.v > w/2.
        
        # This is much faster using numpy!
        
        X = np.array(candidates, dtype=np.int8)
        # Compute Gram matrix
        # X is N x D
        # G = X @ X.T is N x N
        # This is O(N^2 D).
        # For N=5280, D=11. 5280^2 * 11 ~ 300M ops.
        # Numpy can handle this in ~1s.
        
        G = X @ X.T
        # Diagonal is w.
        # Conflict if G[i,j] > w/2.
        # Note: G[i,j] is dot product.
        # For w=3, conflict if dot > 1.5 => dot >= 2.
        # For w=4, conflict if dot > 2.0 => dot >= 3.
        
        # Create conflict mask
        # We want to find independent set.
        # conflict_mask[i,j] = True if G[i,j] > w/2.
        
        threshold = w / 2.0
        # We need strict inequality for conflict?
        # dist^2 >= w.
        # 2w - 2 dot >= w => w >= 2 dot => dot <= w/2.
        # So conflict if dot > w/2.
        
        # For integer dot products:
        # w=3: dot > 1.5 => dot >= 2.
        # w=4: dot > 2.0 => dot >= 3.
        
        # We can create a boolean matrix.
        # But we don't need to store it if we do greedy with numpy slicing?
        # Greedy:
        # available = np.ones(n_cands, dtype=bool)
        # selected = []
        # while available.any():
        #    # Pick an index
        #    # To be fast, pick the first one? Or random?
        #    # Random restarts help.
            
        # Let's do random restarts.
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random order
            order = np.random.permutation(n_cands)
            available = np.ones(n_cands, dtype=bool)
            selected_indices = []
            
            # We can vectorize the removal?
            # For each step, we pick an index u.
            # We need to remove u and all j where conflict(u,j).
            # conflict(u,j) is G[u,j] > threshold.
            
            # This is a sequential process, hard to vectorize fully.
            # But N is small enough.
            
            for u in order:
                if not available[u]:
                    continue
                
                selected_indices.append(u)
                
                # Find neighbors of u that are still available
                # neighbors = np.where((G[u] > threshold) & available)[0]
                # This is O(N) per step.
                # Total O(N^2) per restart.
                # 50 restarts * 14M = 700M ops. Might be too slow.
                
                # Optimization:
                # We can pre-compute the conflict lists for each u?
                # conflict_lists[u] = np.where(G[u] > threshold)[0]
                # This takes O(N^2) memory and time once.
                
            # Let's pre-compute conflict lists once per weight.
            
        # Pre-compute conflict lists
        # This is O(N^2) time and memory.
        # N=5280. 27M ints. ~200MB. OK.
        
        conflict_lists = []
        for i in range(n_cands):
            # G[i] is the row.
            # We want indices j where G[i,j] > threshold.
            # Exclude i itself? G[i,i]=w. w > w/2 is always true.
            # So i is in the list. We should remove it or handle it.
            # We will remove i from available anyway.
            
            # Using numpy where
            idxs = np.where(G[i] > threshold)[0]
            conflict_lists.append(idxs)
            
        # Now run greedy restarts
        for r in range(100): # More restarts possible now
            if time.time() > deadline - 5:
                break
                
            order = np.random.permutation(n_cands)
            available = np.ones(n_cands, dtype=bool)
            selected_indices = []
            
            for u in order:
                if not available[u]:
                    continue
                
                selected_indices.append(u)
                
                # Mark u and its neighbors as unavailable
                # neighbors = conflict_lists[u]
                # available[neighbors] = False
                # available[u] = False (already in neighbors)
                
                # Vectorized update
                available[conflict_lists[u]] = False
            
            if len(selected_indices) > best_score:
                best_score = len(selected_indices)
                best_vectors = [candidates[i] for i in selected_indices]
                
    # If we didn't find anything better than seed (220), return seed?
    # But w=3 should find > 220.
    
    if not best_vectors:
        # Fallback to seed logic
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
