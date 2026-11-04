"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(p)
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _delta_2swap(a, b, p, i, j):
    """
    Compute exact cost change when swapping positions i and j in permutation p.
    p maps facility -> location.
    """
    pi = p[i]
    pj = p[j]
    
    # Precompute the b rows/cols indexed by p
    b_pi_p = b[pi, p]
    b_p_pi = b[p, pi]
    b_pj_p = b[pj, p]
    b_p_pj = b[p, pj]
    
    # Old contributions involving i or j
    # For facility i (at location pi):
    #   row: sum_v a[i,v] * b[pi, p[v]]
    #   col: sum_u a[u,i] * b[p[u], pi]
    old_i_row = np.dot(a[i, :], b_pi_p)
    old_i_col = np.dot(a[:, i], b_p_pi)
    
    # For facility j (at location pj):
    old_j_row = np.dot(a[j, :], b_pj_p)
    old_j_col = np.dot(a[:, j], b_p_pj)
    
    # New contributions after swap:
    # Facility i now at pj:
    new_i_row = np.dot(a[i, :], b_pj_p)
    new_i_col = np.dot(a[:, i], b_p_pj)
    
    # Facility j now at pi:
    new_j_row = np.dot(a[j, :], b_pi_p)
    new_j_col = np.dot(a[:, j], b_p_pi)
    
    # Cross terms (i,j) and (j,i) are double-counted in row+col sums
    old_cross = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    new_cross = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    
    # Diagonal terms (i,i) and (j,j) are double-counted
    old_diag = a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj]
    new_diag = a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]
    
    old_total = old_i_row + old_i_col + old_j_row + old_j_col - old_cross - old_diag
    new_total = new_i_row + new_i_col + new_j_row + new_j_col - new_cross - new_diag
    
    return new_total - old_total


def _spectral_init(a, b, n):
    """
    Spectral initialization using eigenvalue decomposition.
    Align the principal eigenvectors of a and b to get an initial assignment.
    """
    # Symmetrize for spectral methods (QAP often has symmetric matrices)
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get top k eigenvectors
    k = min(n, 20)
    
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_sym)
        
        # Take top k eigenvectors (largest eigenvalues)
        idx_a = np.argsort(-eigvals_a)[:k]
        idx_b = np.argsort(-eigvals_b)[:k]
        
        V_a = eigvecs_a[:, idx_a]
        V_b = eigvecs_b[:, idx_b]
        
        # Procrustes alignment: find orthogonal Q that minimizes ||V_a - V_b Q||_F
        # Q = V_b^T V_a (U S V^T from SVD of V_b^T V_a)
        M = V_b.T @ V_a
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
        
        # Align: X = V_a, Y = V_b Q
        # Assignment: for each facility i, find location j that minimizes distance
        # Use Hungarian on the alignment matrix
        aligned = V_a @ Q.T  # n x n
        
        # Actually, let's use a simpler approach:
        # Project facilities onto the eigenbasis of locations
        # facility_scores[i, j] = sum_k V_a[i,k] * V_b[j,k]
        # Then use Hungarian
        score_matrix = V_a @ V_b.T
        
        # For minimization, we want to maximize alignment, so negate
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
    except Exception:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _wl_color_refinement_init(a, b, n, num_rounds=5):
    """
    Weisfeiler-Lehman color refinement to derive an initial permutation.
    Iteratively refines node colors based on neighborhood structure.
    """
    # Create adjacency-like matrices from a and b
    # For facility graph: weight = a[i,j]
    # For location graph: weight = b[i,j]
    
    # Initialize colors
    colors_a = np.arange(n, dtype=np.int64)  # facility colors
    colors_b = np.arange(n, dtype=np.int64)  # location colors
    
    for _ in range(num_rounds):
        # Update colors based on neighbor color histograms
        new_colors_a = np.zeros(n, dtype=np.int64)
        new_colors_b = np.zeros(n, dtype=np.int64)
        
        for i in range(n):
            # Collect neighbor colors weighted by a[i, :]
            neighbor_weights = a[i, :]
            neighbor_colors = colors_a
            # Create a hash-like signature
            sig_a = np.sum(neighbor_weights[:, None] * (1 + neighbor_colors[None, :]) ** 2, axis=0)
            new_colors_a[i] = hash((i, tuple(sig_a[:10]))) % (2**31)
        
        for j in range(n):
            neighbor_weights = b[j, :]
            neighbor_colors = colors_b
            sig_b = np.sum(neighbor_weights[:, None] * (1 + neighbor_colors[None, :]) ** 2, axis=0)
            new_colors_b[j] = hash((j, tuple(sig_b[:10]))) % (2**31)
        
        # Relabel to compact integers
        unique_a = np.unique(new_colors_a)
        color_map_a = {c: idx for idx, c in enumerate(unique_a)}
        colors_a = np.array([color_map_a[c] for c in new_colors_a], dtype=np.int64)
        
        unique_b = np.unique(new_colors_b)
        color_map_b = {c: idx for idx, c in enumerate(unique_b)}
        colors_b = np.array([color_map_b[c] for c in new_colors_b], dtype=np.int64)
    
    # Match facilities to locations based on final colors
    # For each facility color, find the best matching location color
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Sort facilities by color, then assign to available locations with similar color
    fac_order = np.argsort(colors_a, kind="stable")
    for i in fac_order:
        target_color = colors_a[i]
        # Find unused location with closest color
        candidates = np.where(~used)[0]
        if len(candidates) == 0:
            p[i] = i
            continue
        # Prefer locations with same color, then nearest
        same_color = candidates[colors_b[candidates] == target_color]
        if len(same_color) > 0:
            # Pick the one with minimum distance to facility i's neighborhood profile
            scores = np.array([np.sum(a[i, :] * b[j, :]) for j in same_color])
            best_j = same_color[np.argmin(scores)]
        else:
            scores = np.array([np.sum(a[i, :] * b[j, :]) for j in candidates])
            best_j = candidates[np.argmin(scores)]
        p[i] = best_j
        used[best_j] = True
    
    return p


def _rank_matching_init(a, b, n):
    """Simple rank-matching initialization."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search(a, b, p, deadline, max_rounds=10):
    """
    2-swap local search with time limit.
    Returns (best_p, best_cost).
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # First pass: find all improving swaps
        swaps = []
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                delta = _delta_2swap(a, b, p, i, j)
                if delta < 0:
                    swaps.append((i, j, delta))
        
        if not swaps:
            break
        
        # Apply swaps in order of most improvement first
        swaps.sort(key=lambda x: x[2])
        
        for i, j, delta in swaps:
            if time.time() > deadline - 5:
                break
            # Re-check if this swap is still improving (other swaps may have changed things)
            new_delta = _delta_2swap(a, b, p, i, j)
            if new_delta < 0:
                p[i], p[j] = p[j], p[i]
                current_cost += new_delta
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        if not improved:
            break
    
    return best_p, best_cost


def _random_perturbation(p, n, num_swaps=5):
    """Create a perturbed version of p by swapping random pairs."""
    p_new = p.copy()
    for _ in range(num_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140  # safety margin (150s limit)
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Try multiple initialization strategies
    init_strategies = [
        _spectral_init,
        _wl_color_refinement_init,
        _rank_matching_init,
    ]
    
    for init_fn in init_strategies:
        if time.time() > deadline - 20:
            break
        try:
            p_init = init_fn(a, b, n)
            if p_init is None or len(p_init) != n:
                continue
            # Verify it's a valid permutation
            if len(set(p_init.tolist())) != n:
                continue
            
            p_local, cost_local = _local_search(a, b, p_init, deadline)
            if cost_local < best_cost:
                best_cost = cost_local
                best_p = p_local.copy()
        except Exception:
            continue
    
    # If we have time, do additional random restarts from the best solution
    if best_p is not None and time.time() < deadline - 25:
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            # Random perturbation
            p_rand = _random_perturbation(best_p, n, num_swaps=3 + r)
            p_local, cost_local = _local_search(a, b, p_rand, deadline)
            if cost_local < best_cost:
                best_cost = cost_local
                best_p = p_local.copy()
    
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n, dtype=np.int64)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
