"""Kissing number in dimension 11: construct nonzero integer vectors with min pairwise distance >= max norm."""
import time
import numpy as np

D = 11
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import itertools
import random
import copy
import sys

def _max_norm_and_min_dist(vecs):
    """Compute max norm and min pairwise distance for a list of integer vectors."""
    if len(vecs) < 2:
        return 0, float('inf')
    norms = np.array([np.dot(v, v) for v in vecs], dtype=np.float64)
    max_norm = norms.max()
    # Use matrix for pairwise distances (squared)
    V = np.array(vecs, dtype=np.float64)
    # ||vi - vj||^2 = ||vi||^2 + ||vj||^2 - 2*vi.vj
    G = V @ V.T
    sq_norms = np.diag(G).copy()
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * G
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(dist_sq.min())
    return max_norm, min_dist

def _is_valid(vecs, tol=1e-9):
    """Check if min pairwise distance >= max norm."""
    if len(vecs) < 2:
        return True
    V = np.array(vecs, dtype=np.float64)
    G = V @ V.T
    sq_norms = np.diag(G).copy()
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * G
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = dist_sq.min()
    max_norm_sq = sq_norms.max()
    return min_dist_sq >= max_norm_sq - tol

def _base_ee_vectors():
    """Generate the classical EE lattice kissing vectors in R^11.
    
    The kissing number of 11 is 500 (known result).
    We use the construction from the E8+3 lattice approach or known configurations.
    
    Actually, let's use a simpler approach: generate candidate vectors and filter.
    """
    # Known: kissing number in 11 dimensions is 500
    # Let's construct vectors with norm 2 (squared norm 4) and pairwise distance >= 2
    
    # Strategy: use vectors with entries in {-1, 0, 1} with exactly 2 non-zero entries
    # This gives C(11,2) * 4 = 55 * 4 = 220 vectors with norm 2
    # But we need 500, so we need more.
    
    # Let's use vectors with entries in {-1, 0, 1} with various counts of non-zero entries
    # and check the constraint.
    
    candidates = []
    
    # Type 1: exactly 2 non-zero entries (norm^2 = 2)
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 2: exactly 1 non-zero entry with value 2 (norm^2 = 4)
    for i in range(D):
        for si in (2, -2):
            v = [0] * D
            v[i] = si
            candidates.append(v)
    
    # Type 3: exactly 3 non-zero entries (norm^2 = 3)
    # This might help increase the count
    for idx in range(len(candidates)):
        pass  # Too slow to generate all, let's filter the base set first
    
    return candidates

def _filter_valid_subset(candidates):
    """Find a large valid subset from candidates using greedy + local search."""
    if len(candidates) < 2:
        return list(candidates)
    
    # Sort by norm to prefer smaller norms (easier to satisfy distance constraint)
    sorted_cands = sorted(candidates, key=lambda v: np.dot(v, v))
    
    selected = []
    V_arr = np.array(sorted_cands, dtype=np.float64)
    G = V_arr @ V_arr.T
    sq_norms = np.diag(G).copy()
    
    n = len(sorted_cands)
    # For each candidate, check if it can be added
    # We need: for all selected v_i, ||v_new - v_i|| >= max_norm
    
    # Greedy: add candidates in order of increasing norm
    # Track the max norm so far
    for i in range(n):
        if time.time() > DEADLINE:
            break
        v_new = sorted_cands[i]
        norm_sq_new = sq_norms[i]
        
        # Compute distances to all selected
        if not selected:
            selected.append(i)
            continue
        
        # Check against selected
        valid = True
        # Get indices of selected
        sel_indices = selected
        if len(sel_indices) > 0:
            # dist_sq between v_new and v_sel_j = sq_norms[i] + sq_norms[j] - 2*G[i,j]
            for j in sel_indices:
                dist_sq = sq_norms[i] + sq_norms[j] - 2 * G[i, j]
                if dist_sq < max(sq_norms[sel_indices]) - 1e-9:
                    valid = False
                    break
        
        if valid:
            selected.append(i)
    
    return [sorted_cands[i] for i in selected]

def _local_search(vecs, max_iters=200):
    """Try to improve the configuration by adding/removing/swapping vectors."""
    if not vecs:
        return vecs
    
    V = np.array(vecs, dtype=np.float64)
    n = len(V)
    G = V @ V.T
    sq_norms = np.diag(G).copy()
    max_norm_sq = sq_norms.max()
    
    # Compute min pairwise distance
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * G
    np.fill_diagonal(dist_sq, np.inf)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        iters += 1
        if time.time() > DEADLINE:
            break
        improved = False
        # Try removing a vector that causes issues
        # For each vector, check if removing it allows adding another
        # This is expensive, so just try a few swaps
        # For now, return as-is
        pass
    
    return vecs

def construct():
    start_time = time.time()
    
    # Generate candidate vectors
    candidates = []
    
    # Type 1: exactly 2 non-zero entries from {-1, 1}
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 2: exactly 1 non-zero entry with value ±2
    for i in range(D):
        for si in (2, -2):
            v = [0] * D
            v[i] = si
            candidates.append(v)
    
    # Type 3: exactly 3 non-zero entries from {-1, 1} (norm^2 = 3)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    # Type 4: exactly 2 non-zero entries, one ±2 and one ±1 (norm^2 = 5)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (2, -2):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Filter to find a valid subset
    # Use greedy approach with sorting by norm
    cand_arr = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(cand_arr * cand_arr, axis=1).astype(np.float64)
    
    # Sort by norm
    order = np.argsort(norms_sq)
    sorted_candidates = [candidates[i] for i in order]
    sorted_norms_sq = norms_sq[order]
    
    # Greedy selection
    selected = []
    selected_norms_sq = []
    
    n_cands = len(sorted_candidates)
    
    for i in range(n_cands):
        if time.time() > DEADLINE:
            break
        
        v_new = sorted_candidates[i]
        norm_sq_new = sorted_norms_sq[i]
        
        if not selected:
            selected.append(v_new)
            selected_norms_sq.append(norm_sq_new)
            continue
        
        # Check distance constraint against all selected
        # dist_sq(v_new, v_sel) >= max_norm_sq
        max_norm_sq_current = max(selected_norms_sq)
        
        valid = True
        for idx, v_sel in enumerate(selected):
            diff = np.array(v_new) - np.array(v_sel)
            dist_sq = np.dot(diff, diff)
            if dist_sq < max_norm_sq_current - 1e-9:
                valid = False
                break
        
        if valid:
            selected.append(v_new)
            selected_norms_sq.append(norm_sq_new)
    
    # Convert to array
    result = np.array(selected, dtype=np.int64)
    
    # Verify validity
    if len(result) >= 2:
        V = result.astype(np.float64)
        G = V @ V.T
        sq_norms = np.diag(G).copy()
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * G
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = dist_sq.min()
        max_norm_sq = sq_norms.max()
        if min_dist_sq < max_norm_sq - 1e-9:
            # Invalid, fall back to just the 2-sparse vectors
            fallback = []
            for i in range(D):
                for j in range(i+1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            fallback.append(v)
            result = np.array(fallback, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
