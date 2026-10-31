"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial.distance import cdist

def construct():
    """
    Construct nonzero integer vectors in R^11 using Leech lattice-inspired E8⊕A2⊕A1 structure.
    
    Generates minimal vectors via lattice decomposition, then applies vectorized greedy
    with randomized tie-breaking to find a large independent set in the conflict graph.
    
    For norm^2=4 vectors: need <u,v> <= 2 for all pairs.
    """
    # Generate E8 minimal vectors (240 vectors, norm^2=2) in 8D
    e8_minimal = []
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = si
                v[j] = sj
                e8_minimal.append(v)
    e8_minimal = np.array(e8_minimal, dtype=np.int64)
    
    # Generate A2 minimal vectors (6 vectors, norm^2=2) in 2D
    a2_minimal = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int64)
    
    # Generate A1 minimal vectors (2 vectors, norm^2=1) in 1D
    a1_minimal = np.array([[1], [-1]], dtype=np.int64)
    
    candidates = []
    
    # Type 1: E8 (norm^2=2) ⊕ A2 (norm^2=2) ⊕ [0] → norm^2=4
    for e8v in e8_minimal:
        for a2v in a2_minimal:
            v = np.zeros(D, dtype=np.int64)
            v[:8] = e8v
            v[8:10] = a2v
            candidates.append(v)
    
    # Type 2: E8 (norm^2=2) ⊕ [0,0] ⊕ A1 (norm^2=1) ⊕ one more ±1 → norm^2=4
    for e8v in e8_minimal:
        for a1v in a1_minimal:
            for k in range(D):
                for sign in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[:8] = e8v
                    v[10] = a1v[0]
                    v[k] += sign
                    if np.sum(v**2) == 4:
                        candidates.append(v)
    
    # Type 3: [0,...,0] ⊕ A2 (norm^2=2) ⊕ A1 (norm^2=1) ⊕ one more ±1 → norm^2=4
    for a2v in a2_minimal:
        for a1v in a1_minimal:
            for k in range(D):
                for sign in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[8:10] = a2v
                    v[10] = a1v[0]
                    v[k] += sign
                    if np.sum(v**2) == 4:
                        candidates.append(v)
    
    # Type 4: Four ±1's anywhere (norm^2=4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    # Type 5: One ±2 (norm^2=4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Remove duplicates
    candidates = np.array(candidates, dtype=np.int64)
    candidates = np.unique(candidates, axis=0)
    
    n_candidates = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    # Vectorized greedy with randomization
    np.random.seed(42)
    order = np.random.permutation(n_candidates)
    
    selected_vecs = []
    
    for idx in order:
        c = candidates[idx]
        c_norm = norms_sq[idx]
        
        if not selected_vecs:
            selected_vecs.append(c)
            continue
        
        sel_arr = np.stack(selected_vecs)
        sel_norms = np.array([norms_sq[i] for i in range(len(selected_vecs))])
        # Recompute norms for selected
        sel_norms = np.sum(sel_arr**2, axis=1)
        
        dots = sel_arr @ c
        dists_sq = c_norm + sel_norms - 2 * dots
        
        if np.all(dists_sq >= max_norm_sq):
            selected_vecs.append(c)
    
    result = np.stack(selected_vecs, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
