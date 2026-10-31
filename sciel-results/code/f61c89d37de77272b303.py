"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11
DEADLINE = 120.0  # seconds
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _check_deadline():
    return _elapsed() > DEADLINE - 5.0


# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Score is k (number of vectors).
    
    Strategy: Focus on weight-4 vectors (norm = 2) where the constraint becomes:
    For two weight-4 vectors u, v with ||u||=||v||=2, we need ||u-v|| >= 2,
    which means ||u-v||^2 >= 4, i.e., 8 - 2<u,v> >= 4, so <u,v> <= 2.
    
    This is equivalent to finding a large set of weight-4 vectors with pairwise
    inner products <= 2. We'll use a structured approach with greedy + local search.
    """
    deadline = DEADLINE - 5.0
    
    # Generate all weight-4 vectors: pick 4 coordinates from 11, assign +/-1
    candidates = []
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            candidates.append(v)
    
    candidates_arr = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates_arr)
    
    # All have norm^2 = 4
    # Constraint: ||u-v||^2 >= 4 for all pairs, i.e., <u,v> <= 2
    
    # Precompute inner product matrix (symmetric)
    # This might be large: ~3300 candidates -> 3300x3300 matrix
    # Let's check: C(11,4) * 16 = 330 * 16 = 5280 candidates
    # 5280^2 = ~28M entries, manageable
    
    # Actually, let's be smarter. Use the structure.
    # Two weight-4 vectors u, v:
    # Let S_u, S_v be the support sets (size 4 each)
    # Let |S_u ∩ S_v| = k (0, 1, 2, 3, 4)
    # Inner product = sum over intersection of u_i * v_i
    
    # For a fixed candidate, we can compute compatibility with all others efficiently.
    
    # Greedy approach with randomization
    best_selected = []
    best_count = 0
    
    # Try multiple random orderings
    np.random.seed(42)
    
    for attempt in range(50):
        if _check_deadline():
            break
        
        # Random permutation of candidates
        perm = np.random.permutation(n_candidates)
        
        selected = []
        selected_indices = []
        
        for idx in perm:
            if _check_deadline():
                break
            
            v = candidates_arr[idx]
            
            # Check against all selected
            valid = True
            for sel_idx in selected_indices:
                sel_v = candidates_arr[sel_idx]
                dot = np.dot(v, sel_v)
                if dot > 2:  # inner product must be <= 2
                    valid = False
                    break
            
            if valid:
                selected.append(v)
                selected_indices.append(idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    if not best_selected:
        # Fallback: return all weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
