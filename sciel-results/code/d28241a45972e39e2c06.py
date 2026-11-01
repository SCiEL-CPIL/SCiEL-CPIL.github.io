"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Use a support-disjoint constructive strategy:
    - Partition 11 coordinates into blocks: {0,1,2,3}, {4,5,6,7}, {8,9,10}
    - In each 4-block: all 16 weight-4 vectors (±1 on all 4) + all 8 weight-2 vectors (±2 on one coord) = 24
    - In the 3-block: all 8 weight-3 vectors (±1 on all 3) = 8
    - Vectors from different blocks are orthogonal, so dist^2 = ||v||^2 + ||w||^2 >= 4 + 3 = 7 >= 4 = max_norm^2
    - Within each block, verify pairwise distances >= max_norm^2 = 4
    Total: 24 + 24 + 8 = 56 vectors
    """
    vectors = []
    
    # Block 1: coordinates {0,1,2,3}
    block1_coords = [0, 1, 2, 3]
    # Weight-4 vectors: ±1 on all 4 positions
    for signs in itertools.product([1, -1], repeat=4):
        v = np.zeros(D, dtype=np.int64)
        for pos, sign in zip(block1_coords, signs):
            v[pos] = sign
        vectors.append(v)
    # Weight-2 vectors: ±2 on one coordinate
    for i in block1_coords:
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            vectors.append(v)
    
    # Block 2: coordinates {4,5,6,7}
    block2_coords = [4, 5, 6, 7]
    # Weight-4 vectors: ±1 on all 4 positions
    for signs in itertools.product([1, -1], repeat=4):
        v = np.zeros(D, dtype=np.int64)
        for pos, sign in zip(block2_coords, signs):
            v[pos] = sign
        vectors.append(v)
    # Weight-2 vectors: ±2 on one coordinate
    for i in block2_coords:
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            vectors.append(v)
    
    # Block 3: coordinates {8,9,10}
    block3_coords = [8, 9, 10]
    # Weight-3 vectors: ±1 on all 3 positions
    for signs in itertools.product([1, -1], repeat=3):
        v = np.zeros(D, dtype=np.int64)
        for pos, sign in zip(block3_coords, signs):
            v[pos] = sign
        vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verification: check that min pairwise distance >= max norm
    norms_sq = np.sum(result**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise squared distances
    dots = result @ result.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    np.fill_diagonal(dist_sq, 0)
    min_dist_sq = np.min(dist_sq)
    
    # If verification fails, fall back to greedy search
    if min_dist_sq < max_norm_sq:
        # Fallback: use greedy on all candidates
        candidates = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
        for i in range(D):
            for s in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s * 2
                candidates.append(v)
        for positions in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                candidates.append(v)
        
        candidates = np.array(candidates, dtype=np.int64)
        n = len(candidates)
        norms_sq_c = np.sum(candidates**2, axis=1)
        max_norm_sq_c = 4
        dots_c = candidates @ candidates.T
        dist_sq_c = norms_sq_c[:, None] + norms_sq_c[None, :] - 2 * dots_c
        conflict = (dist_sq_c < max_norm_sq_c)
        np.fill_diagonal(conflict, False)
        
        # Simple greedy
        remaining = np.ones(n, dtype=bool)
        selected = []
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            rem_degrees = conflict[rem_indices].sum(axis=1)
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            chosen_local = min_indices_local[0]
            chosen_global = rem_indices[chosen_local]
            selected.append(chosen_global)
            to_remove = conflict[chosen_global].copy()
            to_remove[chosen_global] = True
            actual_remove = to_remove & remaining
            remaining[actual_remove] = False
        
        if selected:
            return candidates[selected]
        return result
    
    return result
# EVOLVE-BLOCK-END
