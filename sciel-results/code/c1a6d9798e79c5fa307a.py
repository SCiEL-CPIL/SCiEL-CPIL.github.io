import numpy as np
import time
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    timeout = 550  # Safety margin for 600s limit
    
    # Precompute index matrices for all projections
    # P is (10, 10) with P[y, x]
    y_grid, x_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    # Precompute flat indices for each projection
    # r = -1: x - y, range -9..9, shift by 9 -> 0..18
    idx_neg1 = (x_grid - y_grid + 9).flatten().astype(int)
    # r = 0: x, range 0..9
    idx_0 = x_grid.flatten().astype(int)
    # r = 1: x + y, range 0..18
    idx_1 = (x_grid + y_grid).flatten().astype(int)
    # r = 2: x + 2y, range 0..27
    idx_2 = (x_grid + 2 * y_grid).flatten().astype(int)
    
    # Number of bins for each
    n_bins_neg1 = 19
    n_bins_0 = 10
    n_bins_1 = 19
    n_bins_2 = 28
    
    def compute_ratio(P):
        """Fast computation of D/M ratio using precomputed indices."""
        p_flat = P.flatten()
        if not np.all(np.isfinite(p_flat)):
            return -1.0
        p_flat = np.clip(p_flat, 0, None)
        total = np.sum(p_flat)
        if total < 1e-9:
            return -1.0
        p_flat = p_flat / total
        
        # Accumulate counts for each projection using np.add.at
        counts_neg1 = np.zeros(n_bins_neg1)
        np.add.at(counts_neg1, idx_neg1, p_flat)
        
        counts_0 = np.zeros(n_bins_0)
        np.add.at(counts_0, idx_0, p_flat)
        
        counts_1 = np.zeros(n_bins_1)
        np.add.at(counts_1, idx_1, p_flat)
        
        counts_2 = np.zeros(n_bins_2)
        np.add.at(counts_2, idx_2, p_flat)
        
        # Compute entropies
        def entropy_from_counts(counts):
            p = counts[counts > 1e-12]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))
        
        H_neg1 = entropy_from_counts(counts_neg1)
        H_0 = entropy_from_counts(counts_0)
        H_1 = entropy_from_counts(counts_1)
        H_2 = entropy_from_counts(counts_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return -1.0
        return H_neg1 / M

    # Baseline
    P_best = np.ones((n, n)) / (n * n)
    score_best = compute_ratio(P_best)
    
    # Optimization with vectorized ratio computation
    def optimize(P_init, max_iters=50000):
        P = P_init.copy()
        P = np.clip(P, 0, None)
        s = np.sum(P)
        if s < 1e-9: 
            return P, -1
        P = P / s
        
        current_score = compute_ratio(P)
        
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                break
            # Random coordinate descent
            y_idx = random.randint(0, n - 1)
            x_idx = random.randint(0, n - 1)
            
            # Try to increase/decrease P[y_idx, x_idx]
            # We need to maintain sum=1.
            # Simplest: pick another cell to adjust inversely.
            y2 = random.randint(0, n - 1)
            x2 = random.randint(0, n - 1)
            if y_idx == y2 and x_idx == x2:
                continue
                
            # Perturbation
            delta = random.uniform(0.01, 0.1) * max(P[y_idx, x_idx], 1e-6)
            
            # Increase P[y_idx, x_idx] by delta, decrease P[y2, x2] by delta
            P[y_idx, x_idx] += delta
            P[y2, x2] -= delta
            
            # Ensure non-negativity
            if P[y2, x2] < 0:
                P[y2, x2] = 0
                P[y_idx, x_idx] -= delta  # revert
                continue
            
            new_score = compute_ratio(P)
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                P[y_idx, x_idx] -= delta
                P[y2, x2] += delta
        return P, current_score

    # Try several random permutations (sparse diagonal-like)
    best_P = P_best.copy()
    best_score = score_best
    
    # More restarts with more iterations each, enabled by faster compute_ratio
    for _ in range(100):
        if time.time() - start_time > timeout - 10:
            break
        # Random permutation of rows
        perm = np.random.permutation(n)
        P_init = np.zeros((n, n))
        for x in range(n):
            y = perm[x]
            P_init[y, x] = 1.0 / n
        
        P_opt, score = optimize(P_init, max_iters=5000)
        if score > best_score:
            best_score = score
            best_P = P_opt.copy()

    # Try optimizing the uniform distribution itself
    P_opt, score = optimize(P_best, max_iters=10000)
    if score > best_score:
        best_score = score
        best_P = P_opt.copy()

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total < 1e-9:
        return P_best # Fallback
    best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END
