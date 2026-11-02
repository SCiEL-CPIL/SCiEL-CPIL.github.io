"""Kissing number in dimension 11 - structural rewrite using norm-2 vectors."""
import time
import numpy as np
from itertools import combinations, product

D = 11
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _generate_norm2_candidates():
    """
    Generate all integer vectors in R^11 with squared norm = 4.
    These are of two types:
    Type A: ±2e_i (one coordinate is ±2, rest 0) — 22 vectors
    Type B: ±e_i ± e_j ± e_k ± e_l (four coordinates are ±1) — C(11,4)*16 = 330*16 = 5280 vectors
    Total: 5302 candidates.
    
    For norm-squared = 4, the constraint is:
    ||u - v||^2 >= 4 for all pairs.
    ||u - v||^2 = 4 + 4 - 2*u·v = 8 - 2*u·v >= 4
    => u·v <= 2
    
    So we need a set of vectors where every pair has dot product <= 2.
    Since all have norm^2 = 4, this is equivalent to finding a large
    set with max pairwise dot product <= 2.
    """
    candidates = []
    
    # Type A: ±2e_i
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            candidates.append(tuple(v))
    
    # Type B: ±e_i ± e_j ± e_k ± e_l
    for quad in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(quad, signs):
                v[idx] = s
            candidates.append(tuple(v))
    
    return candidates


def _precompute_dot_matrix(candidates):
    """
    Precompute the full dot-product matrix for the candidate pool.
    Returns an int8 array of shape (N, N) where entry (i,j) = candidates[i]·candidates[j].
    """
    n = len(candidates)
    arr = np.array(candidates, dtype=np.int8)
    # Dot product matrix: (N, D) @ (D, N) = (N, N)
    dot_mat = arr @ arr.T
    return dot_mat


def _greedy_max_subset(dot_mat, threshold=2, max_iter=None):
    """
    Greedy: repeatedly add the candidate with the most compatible neighbors.
    A candidate v is compatible with a set S if dot_mat[v, s] <= threshold for all s in S.
    
    We use a bitset-based approach for speed.
    """
    n = dot_mat.shape[0]
    if max_iter is None:
        max_iter = n
    
    # Compatibility matrix: compat[i, j] = True if dot_mat[i,j] <= threshold
    # For same vector, dot = 4 > 2, so self is incompatible (which is fine, we don't add same vector twice)
    compat = (dot_mat <= threshold).astype(np.int8)
    
    # We'll track which candidates are in our selected set
    selected = np.zeros(n, dtype=bool)
    # For each candidate, track which selected ones it's incompatible with
    # Actually, let's use a different approach: maintain a "blocked" count per candidate
    
    # blocked[i] = number of selected vectors incompatible with candidate i
    blocked = np.zeros(n, dtype=np.int32)
    
    # Start with a random valid seed
    # Pick candidate 0 (or any)
    start = 0
    selected[start] = True
    count = 1
    
    # For each other candidate, check if compatible with start
    for j in range(1, n):
        if not compat[start, j]:
            blocked[j] = 1
    
    for _ in range(max_iter - 1):
        # Find candidate with lowest blocked count that is not yet selected
        # and has blocked == 0 (fully compatible)
        # Among fully compatible, pick one that is compatible with the most other unselected candidates
        # to maximize future options
        
        # Candidates with blocked == 0 and not selected
        eligible_mask = (blocked == 0) & (~selected)
        if not np.any(eligible_mask):
            break
        
        eligible_indices = np.where(eligible_mask)[0]
        
        # Score: number of compatible unselected neighbors (to keep options open)
        # This is expensive; use a simpler heuristic: pick the one with most compatible
        # neighbors among all unselected (not just eligible)
        # Actually, let's just pick randomly among eligible for speed, or use a 
        # greedy "most connected" heuristic
        
        # For speed, pick the eligible candidate with the highest degree in the 
        # compatibility graph restricted to unselected nodes
        # This is O(n) per step, total O(n^2) which is ~28M for n=5302 — fine
        
        best_idx = -1
        best_score = -1
        # Sample if too many eligible
        if len(eligible_indices) > 200:
            sample = np.random.choice(eligible_indices, 200, replace=False)
        else:
            sample = eligible_indices
        
        for idx in sample:
            # Count compatible unselected neighbors
            score = np.sum(compat[idx] & (~selected))
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx < 0:
            break
        
        selected[best_idx] = True
        count += 1
        
        # Update blocked counts for all unselected candidates
        # blocked[j] += 1 if not compat[best_idx, j]
        incompatible_mask = ~compat[best_idx]
        blocked[incompatible_mask & (~selected)] += 1
    
    return selected, count


def _local_search(dot_mat, selected, threshold=2, time_budget=10.0):
    """
    Try to improve the selected set by swapping out incompatible candidates
    for better ones, or by finding additional candidates.
    """
    n = dot_mat.shape[0]
    start_time = time.time()
    
    selected_count = np.sum(selected)
    
    # Build list of selected indices
    sel_indices = list(np.where(selected)[0])
    
    # Try to add more candidates that are compatible with all selected
    # blocked[i] = number of selected incompatible with i
    blocked = np.zeros(n, dtype=np.int32)
    for s in sel_indices:
        incompatible = dot_mat[s] > threshold
        blocked[incompatible & (~selected)] += 1
    
    # Try adding candidates with blocked == 0
    for _ in range(50):
        if time.time() - start_time > time_budget:
            break
        eligible = np.where((blocked == 0) & (~selected))[0]
        if len(eligible) == 0:
            break
        # Pick one at random
        idx = eligible[np.random.randint(len(eligible))]
        selected[idx] = True
        selected_count += 1
        incompatible = dot_mat[idx] > threshold
        blocked[incompatible & (~selected)] += 1
    
    return selected, selected_count


def construct():
    start_time = time.time()
    
    # Generate candidates
    candidates = _generate_norm2_candidates()
    n = len(candidates)
    
    # Precompute dot matrix
    dot_mat = _precompute_dot_matrix(candidates)
    
    # Run multiple greedy passes with different random seeds/orderings
    best_selected = None
    best_count = 0
    
    # Pass 1: Standard greedy
    selected, count = _greedy_max_subset(dot_mat, threshold=2)
    if count > best_count:
        best_count = count
        best_selected = selected.copy()
    
    # Pass 2: Greedy with random initialization
    if time.time() - start_time < TIME_LIMIT - 20:
        # Try a few random starts
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            # Random start
            start_idx = np.random.randint(n)
            # Run greedy from this start
            # Modify: force start
            selected_t = np.zeros(n, dtype=bool)
            selected_t[start_idx] = True
            blocked_t = np.zeros(n, dtype=np.int32)
            incompatible = dot_mat[start_idx] > 2
            blocked_t[incompatible & (~selected_t)] += 1
            
            count_t = 1
            for _ in range(n - 1):
                if time.time() - start_time > TIME_LIMIT - 15:
                    break
                eligible_mask = (blocked_t == 0) & (~selected_t)
                if not np.any(eligible_mask):
                    break
                eligible_indices = np.where(eligible_mask)[0]
                if len(eligible_indices) > 100:
                    sample = np.random.choice(eligible_indices, 100, replace=False)
                else:
                    sample = eligible_indices
                
                best_idx = -1
                best_score = -1
                for idx in sample:
                    score = np.sum((dot_mat[idx] <= 2) & (~selected_t))
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx < 0:
                    break
                selected_t[best_idx] = True
                count_t += 1
                incompatible = dot_mat[best_idx] > 2
                blocked_t[incompatible & (~selected_t)] += 1
            
            if count_t > best_count:
                best_count = count_t
                best_selected = selected_t.copy()
    
    # Local search to try to improve
    if time.time() - start_time < TIME_LIMIT - 5 and best_selected is not None:
        best_selected, best_count = _local_search(
            dot_mat, best_selected, threshold=2, 
            time_budget=min(10.0, TIME_LIMIT - (time.time() - start_time) - 3)
        )
    
    # Extract the selected vectors
    selected_indices = np.where(best_selected)[0]
    result = np.array([candidates[i] for i in selected_indices], dtype=np.int64)
    
    # Verify the result
    if len(result) > 0:
        # Check all pairwise distances
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Distance matrix
        dots = result @ result.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, 0)
        min_dist_sq = np.min(dist_sq)
        # Should have min_dist_sq >= max_norm_sq
        # (This is a sanity check; if it fails, something went wrong)
    
    return result
