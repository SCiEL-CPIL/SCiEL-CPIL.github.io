import time
import numpy as np
from itertools import combinations

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # 1. Generate candidate vectors
    # We will consider weights 2 and 3.
    # Norm squared 2: vectors with two 1s or -1s.
    # Norm squared 3: vectors with three 1s or -1s.
    
    candidates = []
    
    # Generate Weight 2 vectors
    # C(11, 2) * 2^2 = 55 * 4 = 220
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Generate Weight 3 vectors
    # C(11, 3) * 2^3 = 165 * 8 = 1320
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We want to maximize k.
    # Constraint: min |u-v| >= max |w|.
    # Let R = max |w|^2.
    # If we only use wt2, R=2.
    # If we use wt2 and wt3, R=3.
    # If we use wt4, R=4.
    
    # Strategy: Try to pack a mix of wt2 and wt3.
    # We will use a greedy algorithm with random restarts.
    
    best_set = []
    best_size = 0
    
    # Heuristic: Process vectors with lower norm first? Or higher?
    # Lower norm vectors are "smaller" and might fit better?
    # Actually, wt2-wt2 constraint is tighter.
    # Let's try sorting by norm squared, then random shuffle.
    
    indices = np.arange(n_cand)
    
    # We'll run several random restarts
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        shuffled_idx = rng.permutation(indices)
        
        current_set = []
        current_vectors = []
        
        # For efficiency, we can maintain the max norm squared in the current set
        # But since we mix wt2 and wt3, R will be 3 if any wt3 is present.
        # If we start with wt2, R=2. Adding a wt3 changes R to 3.
        # This changes the constraints for ALL existing pairs!
        # This makes dynamic insertion tricky.
        
        # Alternative: Decide upfront whether to use wt3 or not.
        # Case 1: Only wt2. R=2.
        # Case 2: Mix wt2 and wt3. R=3.
        
        # Let's try Case 2 first, as it has more candidates.
        # If we commit to R=3, the constraints are:
        # u, v in V: |u-v|^2 >= 3.
        
        # Let's filter candidates to only those with norm squared <= 3?
        # We already only have 2 and 3.
        
        # Greedy insertion with fixed R=3
        # We need to check |u-v|^2 >= 3 for all pairs.
        
        # Optimization: 
        # Precompute dot products? 1540x1540 is ~2.4M entries. Manageable.
        # But we can do it incrementally.
        
        # Let's try a different approach: 
        # Since R is determined by the max norm in the set, and we only have 2 and 3,
        # if we include any wt3, R=3.
        # If we include only wt2, R=2.
        
        # Let's run a greedy for R=3.
        
        # To speed up, we can group by support? No, just brute force with numpy.
        
        # We'll maintain a list of selected vectors.
        # For each new candidate, check against all selected.
        # 1540 * 1540 / 2 ~ 1.2M checks per restart.
        # 50 restarts * 1.2M * 11 ops ~ 660M ops. Might be too slow in Python.
        
        # We need a faster check.
        # Use matrix multiplication?
        # If we select vectors one by one, we can't easily use batched matmul for the new one against the growing set without rebuilding matrices.
        
        # Better: Use a bitset or integer representation?
        # Or just limit the restarts and use a smarter ordering.
        
        # Let's try sorting candidates by "popularity" (how many conflicts they have).
        # Vectors with fewer conflicts are "better" to add first.
        
        # Precompute conflict matrix for the whole pool?
        # 1540x1540 boolean matrix.
        # Memory: 1540*1540 bytes ~ 2.4 MB. Fine.
        
        # Compute |u-v|^2 for all pairs
        # |u-v|^2 = |u|^2 + |v|^2 - 2 u.v
        # We need |u-v|^2 >= 3.
        
        # Let's compute the full compatibility matrix once.
        # This is the "different strategy": Precompute global compatibility, then do Greedy Maximal Independent Set (GMIS) on the compatibility graph.
        
        # Wait, the current program does MIS on *supports*.
        # Doing GMIS on *vectors* is different.
        
        # Let's do it.
        
        # 1. Compute dot products
        # dots = candidates @ candidates.T
        # This is 1540x1540.
        
        if not hasattr(construct, '_compat_cache'):
            # Compute once
            dots = candidates @ candidates.T
            # norms_sq is (N, 1)
            dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
            # Compatibility: dist_sq >= 3
            # Note: dist_sq[i,i] = 0. We don't care about self-compatibility.
            compat = dist_sq >= 3
            # Set diagonal to False to avoid self-selection issues in logic, though we handle indices
            np.fill_diagonal(compat, False)
            construct._compat_cache = compat
        
        compat = construct._compat_cache
        
        # Greedy MIS with random ordering
        # We want to pick a subset of indices such that all pairs are compatible.
        
        # Algorithm:
        # available = set(all indices)
        # while available:
        #    pick v from available (random or heuristic)
        #    add v to set
        #    remove v and all incompatible nodes from available
        
        # To make it faster, we can use bitmasks if N <= 64, but N=1540.
        # We will use numpy arrays for "alive" status.
        
        alive = np.ones(n_cand, dtype=bool)
        selected = []
        
        # Random shuffle of indices to break symmetry
        order = rng.permutation(n_cand)
        
        for v in order:
            if not alive[v]:
                continue
                
            # Select v
            selected.append(v)
            
            # Update alive: remove v and all u such that not compat[u, v]
            # Incompatible with v: ~compat[:, v]
            # But we only care about those currently alive.
            # new_alive = alive & compat[:, v]  (since compat[v,v] is False, v is removed)
            # Wait, compat[:, v] is True for compatible nodes.
            # We want to KEEP compatible nodes.
            # So alive = alive & compat[:, v]
            
            # This operation is O(N) per selection.
            # Total O(N^2) per restart.
            # 1540^2 ~ 2.4M ops.
            # 50 restarts ~ 120M ops.
            # Numpy vectorized ops are fast. This should run in seconds.
            
            alive = alive & compat[:, v]
            
            # Early stop if no alive
            if not np.any(alive):
                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Convert best_set indices to vectors
    if best_size > 0:
        result = candidates[best_set]
    else:
        # Fallback: return just one vector
        result = candidates[:1]
        
    return result

# EVOLVE-BLOCK-END
