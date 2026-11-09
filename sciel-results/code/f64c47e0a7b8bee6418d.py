import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Use a discrete combinatorial greedy set-cover approach.
    2. Treat the support of P as a set of cells.
    3. Greedily add cells that maximize the gain in H(x-y) - max(H(x), H(x+y), H(x+2y)).
    4. Refine the weights on the selected support using local optimization.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 620  # 10s safety buffer

    # Pre-compute indices for entropy calculation
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    best_score = -np.inf
    best_P = np.ones((n, n)) / N # Fallback uniform

    # --- Phase 1: Discrete Greedy Set-Cover ---
    # We want to select a support S such that the distribution of x-y is as uniform as possible
    # while the distributions of x, x+y, x+2y are as concentrated as possible (low entropy).
    # Actually, we want to maximize D/M.
    # D is maxed when x-y is uniform.
    # M is maxed when x, x+y, or x+2y is uniform.
    # So we want x-y uniform, and x, x+y, x+2y NOT uniform (concentrated).
    
    # Let's try to construct a support where x-y is uniform.
    # The diagonals d = x-y+9 have lengths: 1, 2, ..., 9, 10, 9, ..., 1.
    # To make x-y uniform, we should assign equal total weight to each diagonal.
    # Within a diagonal, how should we distribute weight?
    # We want to minimize H(x), H(x+y), H(x+2y).
    # H(x) is entropy of x. To minimize, concentrate x.
    # H(x+y) is entropy of x+y. To minimize, concentrate x+y.
    # H(x+2y) is entropy of x+2y. To minimize, concentrate x+2y.
    
    # If we pick cells such that x is constant, H(x)=0. But then x-y = c-y, so H(x-y) = H(y).
    # If x is constant, x+y = c+y, so H(x+y) = H(y).
    # x+2y = c+2y, so H(x+2y) = H(2y) = H(y) (if y uniform).
    # So D = H(y), M = H(y). Ratio = 1.
    
    # What if we pick cells such that x+y is constant?
    # Then H(x+y) = 0.
    # x-y = x - (c-x) = 2x - c. So H(x-y) = H(2x) = H(x).
    # x = x. So H(x) = H(x).
    # x+2y = x + 2(c-x) = 2c - x. So H(x+2y) = H(x).
    # So D = H(x), M = H(x). Ratio = 1.
    
    # We need a structure where D is large and M is small.
    # D is H(x-y). M is max(H(x), H(x+y), H(x+2y)).
    # To get ratio > 1, we need H(x-y) > max(H(x), H(x+y), H(x+2y)).
    
    # Let's try a greedy search over subsets of cells.
    # Start with empty set.
    # At each step, add the cell that maximizes the gain in D - M.
    # But D and M are entropies, which are submodular/supermodular?
    # Entropy is submodular. So D is submodular. M is max of submodular, which is not necessarily submodular.
    # But let's try greedy anyway.
    
    # Initialize counts
    counts_dm1 = np.zeros(bin_sizes['dm1'])
    counts_x = np.zeros(bin_sizes['x'])
    counts_d1 = np.zeros(bin_sizes['d1'])
    counts_d2 = np.zeros(bin_sizes['d2'])
    
    support = np.zeros(N, dtype=bool)
    P_flat = np.zeros(N)
    
    # We will add cells one by one.
    # To make D large, we want to cover all 19 diagonals.
    # To make M small, we want to concentrate x, x+y, x+2y.
    
    # Let's try to select cells such that x is concentrated.
    # Pick x = 0. Then cells are (y, 0) for y in 0..9.
    # x-y = -y. So x-y takes values 0, -1, ..., -9.
    # x+y = y. So x+y takes values 0, 1, ..., 9.
    # x+2y = 2y. So x+2y takes values 0, 2, ..., 18.
    # If we pick all (y, 0), then H(x) = 0.
    # H(x-y) = H(y) = log2(10).
    # H(x+y) = H(y) = log2(10).
    # H(x+2y) = H(2y) = log2(10).
    # So D = log2(10), M = log2(10). Ratio = 1.
    
    # What if we pick cells such that x-y is uniform, but x is not?
    # We need to mix diagonals.
    
    # Let's use the greedy algorithm to find a good support.
    # We will try to add cells that increase D - M.
    
    # To speed up, we will precompute the contribution of each cell to the counts.
    # But since we are adding cells one by one, we can just update the counts.
    
    # Let's try a few different starting points for the greedy search.
    # Start 1: Empty set.
    # Start 2: Set of cells with x=0.
    # Start 3: Set of cells with y=0.
    # Start 4: Set of cells with x+y=0.
    
    for start_type in ['empty', 'x0', 'y0', 'xy0']:
        counts_dm1 = np.zeros(bin_sizes['dm1'])
        counts_x = np.zeros(bin_sizes['x'])
        counts_d1 = np.zeros(bin_sizes['d1'])
        counts_d2 = np.zeros(bin_sizes['d2'])
        support = np.zeros(N, dtype=bool)
        
        if start_type == 'x0':
            for y in range(n):
                idx = y * n + 0
                support[idx] = True
                counts_dm1[idx_dm1[idx]] += 1
                counts_x[idx_x[idx]] += 1
                counts_d1[idx_d1[idx]] += 1
                counts_d2[idx_d2[idx]] += 1
        elif start_type == 'y0':
            for x in range(n):
                idx = 0 * n + x
                support[idx] = True
                counts_dm1[idx_dm1[idx]] += 1
                counts_x[idx_x[idx]] += 1
                counts_d1[idx_d1[idx]] += 1
                counts_d2[idx_d2[idx]] += 1
        elif start_type == 'xy0':
            for x in range(n):
                y = -x
                if 0 <= y < n:
                    idx = y * n + x
                    support[idx] = True
                    counts_dm1[idx_dm1[idx]] += 1
                    counts_x[idx_x[idx]] += 1
                    counts_d1[idx_d1[idx]] += 1
                    counts_d2[idx_d2[idx]] += 1
        
        # Greedily add cells
        for step in range(N):
            if time.time() > deadline - 10:
                break
            
            best_gain = -np.inf
            best_idx = -1
            
            # Try adding each cell not in support
            for idx in range(N):
                if support[idx]:
                    continue
                
                # Calculate new counts
                new_counts_dm1 = counts_dm1.copy()
                new_counts_x = counts_x.copy()
                new_counts_d1 = counts_d1.copy()
                new_counts_d2 = counts_d2.copy()
                
                new_counts_dm1[idx_dm1[idx]] += 1
                new_counts_x[idx_x[idx]] += 1
                new_counts_d1[idx_d1[idx]] += 1
                new_counts_d2[idx_d2[idx]] += 1
                
                # Calculate new D and M
                D_new = entropy_from_counts(new_counts_dm1)
                M_new = max(entropy_from_counts(new_counts_x), 
                            entropy_from_counts(new_counts_d1), 
                            entropy_from_counts(new_counts_d2))
                
                if M_new < 1e-9:
                    continue
                    
                score_new = D_new / M_new
                
                if score_new > best_gain:
                    best_gain = score_new
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            # Add the best cell
            support[best_idx] = True
            counts_dm1[idx_dm1[best_idx]] += 1
            counts_x[idx_x[best_idx]] += 1
            counts_d1[idx_d1[best_idx]] += 1
            counts_d2[idx_d2[best_idx]] += 1
            
            # Check if we should stop
            if best_gain < 1.0:
                break
        
        # Construct P from support
        P_flat = np.zeros(N)
        P_flat[support] = 1.0
        score = evaluate_P(P_flat)
        
        if score > best_score:
            best_score = score
            best_P = P_flat.reshape((n, n))
            
    # --- Phase 2: Local Refinement on Full Simplex from Best Structural Result ---
    # Use the best structural P as a starting point for full 100-dim L-BFGS-B.
    # This combines the global structural insight with local fine-tuning.
    
    if best_score > -np.inf:
        init_log_p = np.log(best_P.flatten() + 1e-10)
        
        def score_from_log_probs(log_p):
            if time.time() > deadline:
                return 1e9
            log_p_shifted = log_p - np.max(log_p)
            p = np.exp(log_p_shifted)
            s = p.sum()
            if s < 1e-15:
                return 1e9
            p = p / s
            score = evaluate_P(p)
            if not np.isfinite(score):
                return 1e9
            return -score

        try:
            res = minimize(score_from_log_probs, init_log_p, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            if res.fun < 1e8:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    log_p_shifted = res.x - np.max(res.x)
                    p = np.exp(log_p_shifted)
                    p = p / p.sum()
                    best_P = p.reshape((n, n))
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_P)) or best_P.shape != (n, n):
        return np.ones((n, n)) / N
        
    # Clip and normalize for final output
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        return np.ones((n, n)) / N

    return best_P

if __name__ == "__main__":
    pass
