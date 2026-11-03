import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precompute indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # 2. Build NumPy Matrix M (num_k x num_t)
    M = np.zeros((num_k, num_t), dtype=np.uint8)
    for i, block in enumerate(k_blocks):
        for combo in itertools.combinations(block, t):
            M[i, t_index_map[combo]] = 1
            
    # Also keep block_masks as ints for potential fallback or faster bit ops if needed?
    # Actually, for removal, M is good.
    
    # Precompute subset indices for each block for faster removal logic if not using M directly
    # M[i] == 1 gives the indices. np.where(M[i] == 1) is fast.
    
    def _greedy_pass(seed):
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = np.ones(num_t, dtype=np.uint8)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            # Calculate gains
            gains = M.dot(uncovered)
            max_gain = gains.max()
            
            if max_gain == 0:
                break
                
            # Find candidates
            candidates = np.where(gains == max_gain)[0]
            
            # Random selection from candidates
            if len(candidates) > 1 and seed % 2 == 0:
                idx = rng.choice(candidates)
            else:
                idx = candidates[0]
                
            selected.append(idx)
            # Update uncovered
            # M[idx] is a vector of 0s and 1s
            # We want to set uncovered[j] = 0 where M[idx, j] == 1
            uncovered[M[idx] == 1] = 0
            
        return selected

    def _remove_redundant(selected_indices):
        if not selected_indices:
            return selected_indices
            
        # counts of how many blocks cover each t-subset
        counts = np.zeros(num_t, dtype=np.int32)
        # Sum rows
        # M[selected_indices] is a submatrix
        counts += M[selected_indices].sum(axis=0)
        
        kept = list(selected_indices)
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            for i in kept:
                # Check if block i is redundant
                # Get indices covered by block i
                covered_idxs = np.where(M[i] == 1)[0]
                # Check if all counts > 1
                if np.all(counts[covered_idxs] > 1):
                    # Redundant
                    counts[covered_idxs] -= 1
                    changed = True
                else:
                    new_kept.append(i)
            if changed:
                kept = new_kept
                
        return kept

    best_blocks = None
    best_len = float('inf')
    
    attempt = 0
    while time.time() < deadline - 5.0:
        selected = _greedy_pass(attempt)
        if selected is None:
            break
            
        selected = _remove_redundant(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
            
        # Early stop if we hit theoretical lower bound or known best?
        if best_len <= 335:
             # Try a few more to see if we can go lower, but don't waste too much time
             if attempt > 10:
                 break
                 
        attempt += 1
        
    if best_blocks is None:
        # Fallback
        selected = _greedy_pass(0)
        if selected:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []
            
    return best_blocks
