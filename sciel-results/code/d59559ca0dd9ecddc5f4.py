"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # We'll build a set of vectors greedily
    # Condition: for all pairs a,b: a.b <= min(||a||^2, ||b||^2) / 2
    
    selected = []  # list of vectors
    selected_norms_sq = []  # ||v||^2
    
    def check_compatible_batch(candidates, norms_sq_list):
        """Check which candidates are compatible with all selected vectors.
        Returns a boolean array."""
        if not selected:
            return np.ones(len(candidates), dtype=bool)
        
        # selected_matrix: (n_selected, D)
        # candidates: (n_candidates, D)
        # dot products: (n_candidates, n_selected)
        sel_mat = np.array(selected, dtype=np.int64)
        cand_mat = np.array(candidates, dtype=np.int64)
        
        # Compute all dot products: (n_candidates, n_selected)
        dots = cand_mat @ sel_mat.T
        
        # For each candidate i and selected j: need dots[i,j] <= min(norms_sq[i], sel_norms[j]) / 2
        # Create min_norms matrix: (n_candidates, n_selected)
        cand_norms = np.array(norms_sq_list, dtype=np.float64)  # (n_candidates,)
        sel_norms = np.array(selected_norms_sq, dtype=np.float64)  # (n_selected,)
        
        min_norms = np.minimum(cand_norms[:, np.newaxis], sel_norms[np.newaxis, :])  # (n_candidates, n_selected)
        threshold = min_norms / 2.0
        
        # Check: dots[i,j] <= threshold[i,j] for all j
        # Use float comparison with small tolerance
        compatible = np.all(dots <= threshold + 1e-9, axis=1)
        
        return compatible
    
    # Generate weight-2 vectors first
    weight2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                weight2_vectors.append(v)
    
    # Add all weight-2 vectors
    for v in weight2_vectors:
        selected.append(v)
        selected_norms_sq.append(2)
    
    # Now try higher weights
    for weight in range(3, D + 1):
        if time.time() > deadline:
            break
        
        # Generate all vectors of this weight
        candidates = []
        for positions in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                candidates.append(v)
        
        if not candidates:
            continue
        
        candidates_arr = np.array(candidates, dtype=np.int64)
        norms_sq_list = [weight] * len(candidates)
        
        # Check compatibility
        compatible = check_compatible_batch(candidates_arr, norms_sq_list)
        
        # Add compatible vectors one by one (greedy, re-checking after each addition)
        # For efficiency, add all compatible ones, but we need to check pairwise among them too
        # Actually, we need to check that the new vectors are also compatible with each other
        
        # Let's do it iteratively: try to add each candidate
        for idx in range(len(candidates)):
            if time.time() > deadline:
                break
            if not compatible[idx]:
                continue
            
            v = candidates_arr[idx]
            v_norm_sq = weight
            
            # Check against all currently selected
            sel_mat = np.array(selected, dtype=np.int64)
            dots = sel_mat @ v
            sel_norms = np.array(selected_norms_sq, dtype=np.float64)
            min_norms = np.minimum(v_norm_sq, sel_norms)
            threshold = min_norms / 2.0
            
            if np.all(dots <= threshold + 1e-9):
                selected.append(v)
                selected_norms_sq.append(v_norm_sq)
                # Update compatibility for remaining candidates
                # This is expensive; let's just recompute for next batch
                # Actually, let's break and regenerate
                break
        
        # If we added any, we need to recheck remaining candidates
        # For simplicity, let's just continue to next weight
        # This might miss some, but let's see if we get enough
    
    if not selected:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    result = np.array(selected, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
