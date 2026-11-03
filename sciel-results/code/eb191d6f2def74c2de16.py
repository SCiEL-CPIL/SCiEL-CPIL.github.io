import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Generate candidates with squared norm 4
    # Type 1: One 2, rest 0: ±2e_i
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    candidates = []
    
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
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
    
    # Build conflict masks
    # conflict_mask[i] has bit j set if M[i,j] > 2
    conflict_masks = [0] * N
    # We only need to check j > i for symmetry, but building full mask is easier with numpy first?
    # Let's use numpy to create a boolean matrix, then convert to int masks.
    # M is int32.
    conflict_bool = M > 2
    # Diagonal is True (M[i,i]=4 > 2), but we don't care about self-conflict in the mask logic 
    # if we ensure we don't add self. But for the check (mask & current) == 0, 
    # if self is in current, it's already added. 
    # Actually, when checking if u can be added, u is NOT in current_mask.
    # So conflict_mask[u] & current_mask checks conflicts with others.
    # We should clear the diagonal bit in conflict_masks to be safe? 
    # No, if u is not in current_mask, bit u in current_mask is 0.
    # So bit u in conflict_mask[u] doesn't matter.
    
    # Convert boolean matrix to list of ints
    # This is the slow part if done naively.
    # We can iterate rows.
    for i in range(N):
        mask = 0
        row = conflict_bool[i]
        # Find indices where row is True
        # This is slow in Python loop.
        # Alternative: use numpy to find non-zero indices?
        # Or just keep the numpy boolean matrix and use numpy for checks?
        # N=5300. N^2 = 28M bools.
        # np.any(conflict_bool[u] & current_bool)
        # This is vectorized.
        
    # Let's stick to numpy boolean arrays for the set and conflicts.
    # It's safer and likely fast enough for 150s.
    
    # Greedy Init
    rng = np.random.default_rng(42)
    best_mask = None
    best_count = 0
    
    # Multiple greedy starts
    for trial in range(20):
        if time.time() > deadline - 10:
            break
            
        if trial == 0:
            # Sort by degree (number of conflicts) ascending
            degrees = np.sum(conflict_bool, axis=1)
            order = np.argsort(degrees, kind='stable')
        else:
            order = rng.permutation(N)
            
        current_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if count == 0:
                current_mask[idx] = True
                count += 1
                continue
                
            # Check if idx conflicts with any in current_mask
            # conflict_bool[idx] & current_mask
            if not np.any(conflict_bool[idx] & current_mask):
                current_mask[idx] = True
                count += 1
                
        if count > best_count:
            best_count = count
            best_mask = current_mask.copy()
            
    # Local Search on the best greedy solution
    if best_mask is not None and time.time() < deadline - 5:
        current_mask = best_mask.copy()
        current_count = best_count
        best_mask_ls = best_mask.copy()
        best_count_ls = best_count
        
        # Precompute list of all indices
        all_indices = np.arange(N)
        
        while time.time() < deadline - 2:
            # Pick a random index from current_mask
            selected_indices = np.where(current_mask)[0]
            if len(selected_indices) == 0:
                break
            
            rem_idx = selected_indices[rng.integers(len(selected_indices))]
            
            # Remove it
            current_mask[rem_idx] = False
            current_count -= 1
            
            # Repair: Try to add any compatible vector
            # We scan all N vectors. 
            # Optimization: Only scan vectors that were previously blocked by rem_idx?
            # Hard to know which ones. Just scan all.
            # To speed up, we can shuffle the scan order or just go linear.
            # Linear scan is deterministic. 
            # Let's try to add vectors in a random order to escape local minima?
            # Or just linear.
            
            added_any = False
            for idx in all_indices:
                if current_mask[idx]:
                    continue
                if not np.any(conflict_bool[idx] & current_mask):
                    current_mask[idx] = True
                    current_count += 1
                    added_any = True
            
            if current_count > best_count_ls:
                best_count_ls = current_count
                best_mask_ls = current_mask.copy()
                
            # Optional: If we added nothing, we might be stuck. 
            # The removal might not have freed anything compatible.
            # We can just continue.
            
    if best_mask_ls is not None:
        return candidates[best_mask_ls].astype(np.int64)
    elif best_mask is not None:
        return candidates[best_mask].astype(np.int64)
    else:
        # Fallback
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

