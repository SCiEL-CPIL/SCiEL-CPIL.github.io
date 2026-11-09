# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import rankdata

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for a discrete measure."""
    n = len(support)
    if n == 0:
        return 0.0
    s = support.astype(np.float64)
    w = weights.astype(np.float64)
    # Ensure weights are normalized for internal computation
    total_w = np.sum(w)
    if total_w <= 0:
        return 0.0
    p = w / total_w
    
    # Compute pairwise sums S_{ij} = s_i + s_j and their joint probabilities P_{ij} = p_i p_j
    S = np.add.outer(s, s)  # Shape (n, n)
    P_pair = np.outer(p, p) # Shape (n, n)
    
    # For each k (X3) and l (X4), we need sum_{i,j} P_{ij} * I(s_i + s_j < 2*s_l - s_k)
    # Let's vectorize over k and l.
    # Threshold T_{kl} = 2*s_l - s_k. Shape (n, n) where first dim is k, second is l.
    # Wait, the formula is sum_{k,l} p_k p_l * (sum_{i,j} p_i p_j I(S_{ij} < 2 s_l - s_k))
    # Let's create a 4D tensor or use broadcasting.
    # s_k is shape (n, 1) if we treat k as first dim?
    # Let's align indices carefully.
    # k index for X3, l index for X4.
    # s_k: shape (n, 1) -> broadcast to (n, n) if we want T[k, l] = 2 s_l - s_k.
    # Let's make s_k column vector (n, 1) and s_l row vector (1, n).
    sk = s[:, np.newaxis]  # (n, 1)
    sl = s[np.newaxis, :]  # (1, n)
    T = 2 * sl - sk        # (n, n) where T[k, l] = 2 s_l - s_k
    
    # Now we need for each (k, l), sum of P_pair where S < T[k, l].
    # S is (n, n). P_pair is (n, n).
    # We can flatten S and P_pair.
    S_flat = S.flatten()
    P_flat = P_pair.flatten()
    
    # For each threshold T_val, we need sum(P_flat[S_flat < T_val]).
    # Since T has shape (n, n), we have n^2 thresholds.
    # We can sort S_flat and use cumulative sum of P_flat.
    # sort_idx = np.argsort(S_flat)
    # S_sorted = S_flat[sort_idx]
    # P_cumsum = np.cumsum(P_flat[sort_idx])
    
    # But doing this n^2 times via searchsorted might be slow if n is large?
    # n is small (up to ~10 or 20). n^2 = 100-400. Searchsorted on array of size n^2 is fast.
    # However, we can vectorize the search.
    
    # Let's try a direct loop over k and l, but vectorized over i, j.
    # Actually, let's just do the searchsorted approach. It's robust.
    
    order = np.argsort(S_flat)
    S_sorted = S_flat[order]
    P_sorted = P_flat[order]
    P_cumsum = np.cumsum(P_sorted)
    
    # For each T in T.flatten(), find index of last element in S_sorted < T.
    # np.searchsorted(S_sorted, T_val, side='right') - 1 gives index of last < T_val.
    # If index is -1, sum is 0.
    
    T_flat = T.flatten()
    # searchsorted returns insertion point. side='right' means index where T_val would be inserted to maintain order,
    # which is the number of elements <= T_val.
    # We want strictly < T_val.
    # So index = np.searchsorted(S_sorted, T_flat, side='left') gives number of elements < T_val?
    # No, side='left' gives index of first element >= T_val. So index is count of elements < T_val.
    # Yes.
    idx = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Now we need P_cumsum[idx - 1]. If idx == 0, sum is 0.
    # We can use np.where or a padded cumsum.
    # Let's pad P_cumsum with 0 at the beginning.
    P_cumsum_padded = np.concatenate(([0.0], P_cumsum))
    sums = P_cumsum_padded[idx]
    
    # Now sums has shape (n, n) corresponding to (k, l).
    # We need sum_{k, l} p_k p_l * sums[k, l].
    P_kl = np.outer(p, p) # p_k p_l
    prob = np.sum(P_kl * sums)
    
    return prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds limit, leave some margin
    safety_margin = 10.0
    
    best_score = 0.0
    best_params = None
    best_n = 1
    best_support = np.array([0.0])
    best_weights = np.array([1.0])
    
    # Analytical solution for N=2
    # Support [0, 1], Weights [p, 1-p]
    # P = p^4 + p^3(1-p) + p^2(1-p)^2 + ... ?
    # Let's just evaluate it.
    s2 = np.array([0.0, 1.0])
    # Optimize p
    def neg_P_2(p):
        w = np.array([p, 1-p])
        return -_compute_P(s2, w)
    
    res2 = differential_evolution(neg_P_2, bounds=[(1e-6, 1-1e-6)], maxiter=100, tol=1e-8, polish=False)
    if res2.fun is not None:
        p_opt = res2.x[0]
        w2 = np.array([p_opt, 1-p_opt])
        score2 = -res2.fun
        if score2 > best_score:
            best_score = score2
            best_params = w2
            best_n = 2
            best_support = s2.copy()
            best_weights = w2.copy()
            
    # Check time
    if time.time() > deadline - safety_margin:
        return best_support, best_weights
    
    # Try N=3
    # Support [0, 1, M]. Weights [p0, p1, p2].
    # Params: p0, p1, M. p2 = 1 - p0 - p1.
    # Constraints: p0>=0, p1>=0, p2>=0, M>=0.
    # We can optimize p0, p1, M.
    # Bounds: p0 in [0,1], p1 in [0, 1-p0], M in [0, UpperBound].
    # Since DE doesn't handle dynamic bounds easily, we can use transformations.
    # Or just fix M to a range and optimize p0, p1.
    # Let's try a few fixed M values first.
    M_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    
    for M in M_values:
        s3 = np.array([0.0, 1.0, M])
        
        def neg_P_3(p01):
            p0 = p01[0]
            p1 = p01[1]
            p2 = 1 - p0 - p1
            if p2 < 0:
                return 1e10
            w = np.array([p0, p1, p2])
            return -_compute_P(s3, w)
        
        # Bounds: p0 in [0, 1], p1 in [0, 1] (but p0+p1 <= 1 is handled by penalty)
        # To be safe, we can use a transformation that ensures p0+p1 <= 1.
        # p0 = x * (1-y), p1 = y * (1-x)? No, simpler:
        # Let u, v be in [0,1].
        # p0 = u * v
        # p1 = (1-u) * v
        # p2 = 1 - v
        # This ensures p0+p1+v=1 and all non-negative.
        
        def neg_P_3_trans(params):
            u = params[0]
            v = params[1]
            p0 = u * v
            p1 = (1-u) * v
            p2 = 1 - v
            w = np.array([p0, p1, p2])
            return -_compute_P(s3, w)
        
        try:
            res3 = differential_evolution(neg_P_3_trans, bounds=[(0, 1), (0, 1)], maxiter=200, tol=1e-8, polish=True, seed=42)
            if res3.fun is not None and res3.fun < 1e9:
                score3 = -res3.fun
                if score3 > best_score:
                    u = res3.x[0]
                    v = res3.x[1]
                    p0 = u * v
                    p1 = (1-u) * v
                    p2 = 1 - v
                    w3 = np.array([p0, p1, p2])
                    best_score = score3
                    best_params = w3
                    best_n = 3
                    best_support = s3.copy()
                    best_weights = w3.copy()
        except Exception:
            pass
            
        if time.time() > deadline - safety_margin:
            return best_support, best_weights

    # Try N=4
    # Support [0, 1, M1, M2] with 1 < M1 < M2.
    # This is getting complicated. Let's stick to simpler structures if time is tight.
    # A common good structure is [0, 1, M] or [0, M].
    
    # Let's also try optimizing M for N=3
    # We already did this by fixing M. Let's do a global search over M, p0, p1.
    # Params: u, v, logM.
    # p0 = u*v, p1 = (1-u)*v, p2 = 1-v.
    # M = exp(logM). logM in [0, 5] (M in [1, 148]).
    
    def neg_P_3_global(params):
        u = params[0]
        v = params[1]
        logM = params[2]
        M = np.exp(logM)
        s3 = np.array([0.0, 1.0, M])
        p0 = u * v
        p1 = (1-u) * v
        p2 = 1 - v
        w = np.array([p0, p1, p2])
        return -_compute_P(s3, w)
    
    try:
        res3g = differential_evolution(neg_P_3_global, bounds=[(0, 1), (0, 1), (0, 6)], maxiter=500, tol=1e-8, polish=True, seed=42)
        if res3g.fun is not None and res3g.fun < 1e9:
            score3g = -res3g.fun
            if score3g > best_score:
                u = res3g.x[0]
                v = res3g.x[1]
                M = np.exp(res3g.x[2])
                p0 = u * v
                p1 = (1-u) * v
                p2 = 1 - v
                w3 = np.array([p0, p1, p2])
                s3 = np.array([0.0, 1.0, M])
                best_score = score3g
                best_params = w3
                best_n = 3
                best_support = s3
                best_weights = w3
    except Exception:
        pass

    # Final check and return
    if time.time() > deadline - safety_margin:
        pass
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
