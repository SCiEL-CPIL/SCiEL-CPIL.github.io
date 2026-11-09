"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def construct():
    # Generate candidate vectors with small integer entries
    # We'll try vectors with entries in {-1, 0, 1} first, then expand
    candidates = []
    
    # Generate all vectors with entries in {-1, 0, 1} with at least 2 non-zero entries
    for entry_tuple in product([-1, 0, 1], repeat=D):
        v = np.array(entry_tuple, dtype=np.int64)
        nnz = np.count_nonzero(v)
        if nnz >= 2:
            candidates.append(v)
    
    # Sort by norm (ascending) to prefer lower-norm vectors
    norms = np.array([np.linalg.norm(v) for v in candidates])
    sorted_indices = np.argsort(norms)
    candidates = [candidates[i] for i in sorted_indices]
    norms = [norms[i] for i in sorted_indices]
    
    # Greedily select vectors satisfying the distance constraint
    selected = []
    selected_norms = []
    
    for v, norm_v in zip(candidates, norms):
        if not selected:
            selected.append(v)
            selected_norms.append(norm_v)
            continue
        
        # Check distance constraint with all previously selected vectors
        valid = True
        for u, norm_u in zip(selected, selected_norms):
            dist = np.linalg.norm(v - u)
            max_norm = max(norm_v, norm_u)
            if dist < max_norm - 1e-9:
                valid = False
                break
        
        if valid:
            selected.append(v)
            selected_norms.append(norm_v)
    
    if not selected:
        # Fallback: return at least one vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
