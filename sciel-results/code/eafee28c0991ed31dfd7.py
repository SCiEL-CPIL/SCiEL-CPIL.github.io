"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140.0  # 10s safety margin
    
    # 1. Generate all candidate vectors with support size 4 and entries +/- 1
    # Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates = []
    indices = range(D)
    for comb in __import__('itertools').combinations(indices, 4):
        # For each combination of 4 indices, generate all 16 sign patterns
        # v is a vector of 0s, with +/- 1 at the indices in comb
        for signs in __import__('itertools').product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # In-mask: boolean array of which candidates are currently in the set
    # We represent the set S as a list of indices into 'candidates'
    # Or better, keep S as a numpy array of the actual vectors for fast dot products?
    # If we keep indices, we have to gather vectors. 
    # Keeping vectors is faster for dot product if we update the array.
    # But swapping requires removing/adding rows. 
    # Let's keep a list of indices for the set, and a boolean mask for "in_set".
    
    in_set = np.zeros(n_candidates, dtype=bool)
    set_indices = []
    
    # Helper to check compatibility of a candidate vector with the current set
    # S is stored as a numpy array of vectors?
    # If we use indices, S_vecs = candidates[set_indices]
    # This copy is O(|S|).
    
    def get_set_vectors():
        if not set_indices:
            return np.empty((0, D), dtype=np.int64)
        return candidates[set_indices]
    
    def is_compatible(v_idx):
        # v_idx is an index into candidates
        # Check if candidates[v_idx] has dot product <= 2 with all in set
        if not set_indices:
            return True
        v = candidates[v_idx]
        # Batch dot product
        # S_vecs = candidates[set_indices]
        # dots = S_vecs @ v
        # return np.all(dots <= 2)
        
        # Optimization: Pre-compute dot products?
        # Since S changes, we can't pre-compute fully.
        # But we can check quickly.
        # For ~600 vectors, this is fast.
        S_vecs = candidates[set_indices]
        dots = S_vecs @ v
        return np.all(dots <= 2)

    best_count = 0
    best_set_indices_copy = []
    
    # Initialize with a greedy pass
    # Randomize order for initial greedy to find a decent base
    initial_order = np.random.permutation(n_candidates)
    for idx in initial_order:
        if is_compatible(idx):
            in_set[idx] = True
            set_indices.append(idx)
            
    if len(set_indices) > best_count:
        best_count = len(set_indices)
        best_set_indices_copy = list(set_indices)
        
    # Local Search / Simulated Annealing-like improvement
    # We will try to add vectors if possible, and if not, perform swaps to rearrange
    
    current_count = len(set_indices)
    
    while time.time() - start_time < timeout:
        improved = False
        
        # 1. Try to add any compatible vector
        # Check all candidates not in set
        # This is O(N * |S|). 5280 * 600 ~ 3M ops. Fast.
        
        # Optimization: Only check candidates that were previously rejected?
        # No, keep it simple first.
        
        # Vectorized check for all candidates not in set?
        # S_vecs shape (k, D). Candidates shape (N, D).
        # Dots shape (k, N).
        # We want to find j such that dots[:, j] <= 2 for all i.
        
        if set_indices:
            S_vecs = candidates[set_indices]
            # Compute all pairwise dot products between S and all candidates
            # This is a matrix multiplication (k x D) * (D x N) -> (k x N)
            dots_all = S_vecs @ candidates.T
            # A candidate j is compatible if max(dots_all[:, j]) <= 2
            max_dots = np.max(dots_all, axis=0)
            
            # Candidates not in set: ~not in_set
            # Candidates compatible: max_dots <= 2
            # Intersection: 
            compatible_mask = (max_dots <= 2) & (~in_set)
            
            if np.any(compatible_mask):
                # Add all compatible vectors? 
                # If we add one, the set changes, so we should check again.
                # But adding all at once might violate constraints between the new vectors?
                # No, the check was against the OLD set.
                # The new vectors might conflict with each other!
                # Example: u and v are both compatible with S, but u.v > 2.
                # So we must add one by one or check mutual compatibility.
                # Given speed, add one by one is safer.
                
                # Pick one randomly from compatible ones to introduce randomness?
                # Or just add the first one.
                # Let's add one, update S, and loop.
                add_indices = np.where(compatible_mask)[0]
                if len(add_indices) > 0:
                    # Random choice to break ties
                    idx_to_add = add_indices[np.random.randint(len(add_indices))]
                    in_set[idx_to_add] = True
                    set_indices.append(idx_to_add)
                    improved = True
                    current_count += 1
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_set_indices_copy = list(set_indices)
                    
                    continue # Restart the scan with new S
        
        # 2. If no additions possible, try a swap
        if current_count > 0:
            # Pick a random vector to remove
            out_idx_pos = np.random.randint(current_count)
            out_idx = set_indices[out_idx_pos]
            
            # Check if any vector can replace it
            # New set S' = S \ {out_idx}
            # We need a v not in S such that v is compatible with S'.
            # Note: v was rejected before because it conflicted with SOME vector in S.
            # If it only conflicted with out_idx, it is now compatible with S'.
            
            if len(set_indices) > 1:
                S_prime_vecs = candidates[[i for i in set_indices if i != out_idx]]
                dots_prime = S_prime_vecs @ candidates.T
                max_dots_prime = np.max(dots_prime, axis=0)
                
                # Compatible with S_prime
                compatible_with_prime = max_dots_prime <= 2
                
                # Must not be in set (except out_idx itself, but we don't want to add it back immediately usually, 
                # though it's harmless. We want a NEW vector).
                # So: compatible_with_prime AND NOT in_set
                swap_mask = compatible_with_prime & (~in_set)
                
                if np.any(swap_mask):
                    in_indices = np.where(swap_mask)[0]
                    in_idx = in_indices[np.random.randint(len(in_indices))]
                    
                    # Perform swap
                    in_set[out_idx] = False
                    set_indices[out_idx_pos] = in_idx
                    in_set[in_idx] = True
                    
                    # Count stays same. No immediate improvement in count, 
                    # but might allow future additions.
                    # We don't update best_count here as it didn't change.
                    # improved = True (to indicate state change) but we don't 'continue' necessarily?
                    # Actually, after a swap, we should re-evaluate additions.
                    # The `continue` in the add branch handles this.
                    # Here we just fall through to the next while iteration.
            
    # Return the best set found
    if best_set_indices_copy:
        return candidates[best_set_indices_copy]
    else:
        # Fallback to seed logic if something failed
        vectors = []
        for i, j in __import__('itertools').combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
