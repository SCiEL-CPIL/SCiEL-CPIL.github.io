# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0


def _deadline_ok(margin=30.0):
    return (time.time() - _START) < (_DEADLINE - margin)


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for given support and weights."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-12:
        return 0.0
    p = w / w_sum
    # P = sum_{k,l} p_k * p_l * Q(2*s_l - s_k)
    # where Q(t) = sum_{i,j} p_i * p_j * I(s_i + s_j < t)
    # Precompute all pairwise sums
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    # For each (k, l), compute Q(2*s_l - s_k)
    # Thresholds: T[k, l] = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]  # (k, l)
    # For each (k, l), count pairs (i, j) with pair_sums[i, j] < T[k, l]
    # G[k, l, i, j] = I(pair_sums[i, j] < T[k, l])
    # Then Q[k, l] = sum_{i,j} p_i * p_j * G[k, l, i, j]
    # P = sum_{k,l} p_k * p_l * Q[k, l]
    
    # Efficient computation using broadcasting
    # G has shape (n, n, n, n) which is n^4 - too large for n > ~100
    # Instead, for each (k, l), we can compute Q more efficiently
    # Sort pair_sums and use cumulative sums
    
    # Alternative: compute Q for each unique threshold
    # But thresholds depend on both k and l, so there are n^2 unique thresholds
    
    # For small n (<= 20), direct computation is fine
    if n <= 20:
        G = (pair_sums[None, None, :, :] < T[:, :, None, None])  # (k, l, i, j)
        W2 = p[:, None] * p[None, :]  # (i, j)
        Q = np.einsum('kl,klij,ij->kl', G, G, W2)  # This is wrong
        # Correct: Q[k,l] = sum_{i,j} p_i * p_j * G[k,l,i,j]
        Q = np.einsum('klij,ij->kl', G, W2)
        P = np.einsum('kl,kl->', p[:, None] * p[None, :], Q)
        return float(P)
    else:
        # For larger n, use a different approach
        # Precompute sorted pair sums and their cumulative weights
        # But this is complex; for now, limit to n <= 20
        return 0.0


def _compute_P_fast(support, weights):
    """Fast computation using the structure of the problem."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-12:
        return 0.0
    p = w / w_sum
    
    # P = E[I(X1+X2+X3 < 2*X4)]
    # = sum_l p_l * P(X1+X2+X3 < 2*s_l)
    # For each l, compute P(X1+X2+X3 < 2*s_l) = sum_{i,j,k} p_i*p_j*p_k * I(s_i+s_j+s_k < 2*s_l)
    
    # Precompute all triple sums
    # triple_sums[i,j,k] = s_i + s_j + s_k
    # This is n^3 which is manageable for n <= 50
    
    if n <= 50:
        # Build triple sum matrix
        triple_sums = s[:, None, None] + s[None, :, None] + s[None, None, :]  # (i, j, k)
        W3 = p[:, None, None] * p[None, :, None] * p[None, None, :]  # (i, j, k)
        
        # For each l, threshold is 2*s_l
        # Q[l] = sum_{i,j,k} W3[i,j,k] * I(triple_sums[i,j,k] < 2*s_l)
        thresholds = 2.0 * s  # (l,)
        # G[l, i, j, k] = I(triple_sums[i,j,k] < thresholds[l])
        G = (triple_sums[None, :, :, :] < thresholds[:, None, None, None])  # (l, i, j, k)
        Q = np.einsum('lijk,ijk->l', G, W3)
        P = np.dot(p, Q)
        return float(P)
    else:
        return 0.0


def _log_grid_search(n_points, log_min, log_max, max_iter=100, step_size=0.1):
    """
    Search over support points on a logarithmic grid using alternating projection.
    
    Strategy:
    1. Initialize support points on a log grid
    2. For fixed support, optimize weights using gradient ascent
    3. For fixed weights, optimize support positions
    4. Repeat until convergence or deadline
    """
    global _START
    
    # Initial support on log grid
    log_support = np.linspace(log_min, log_max, n_points)
    support = np.exp(log_support)
    
    # Initial weights: uniform
    weights = np.ones(n_points) / n_points
    
    best_val = _compute_P_fast(support, weights)
    best_support = support.copy()
    best_weights = weights.copy()
    
    # Alternating optimization
    for it in range(max_iter):
        if not _deadline_ok(margin=60.0):
            break
        
        # Step 1: Optimize weights for fixed support
        # Use gradient ascent with projection onto simplex
        def neg_P(w):
            w = np.clip(w, 0, None)
            w_sum = w.sum()
            if w_sum <= 1e-12:
                return 1.0
            w = w / w_sum
            return -_compute_P_fast(support, w)
        
        # Simple gradient ascent with projection
        lr = 0.05
        for inner in range(50):
            if not _deadline_ok(margin=60.0):
                break
            # Compute gradient numerically
            grad = np.zeros(n_points)
            eps = 1e-6
            for i in range(n_points):
                w_plus = weights.copy()
                w_plus[i] += eps
                w_minus = weights.copy()
                w_minus[i] -= eps
                w_plus = np.clip(w_plus, 0, None)
                w_minus = np.clip(w_minus, 0, None)
                gp = w_plus.sum()
                gm = w_minus.sum()
                if gp > 1e-12 and gm > 1e-12:
                    grad[i] = (neg_P(w_plus/gp) - neg_P(w_minus/gm)) / (2*eps)
            
            # Update
            weights = weights - lr * grad
            weights = np.clip(weights, 0, None)
            w_sum = weights.sum()
            if w_sum > 1e-12:
                weights = weights / w_sum
        
        # Step 2: Optimize support for fixed weights
        # Use coordinate descent on log-support
        for inner in range(30):
            if not _deadline_ok(margin=60.0):
                break
            improved = False
            for i in range(n_points):
                # Try small perturbations
                base_val = _compute_P_fast(support, weights)
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    new_log_support = log_support.copy()
                    new_log_support[i] += delta
                    # Ensure sorted and positive
                    if np.any(new_log_support < np.log(1e-6)):
                        continue
                    new_support = np.exp(new_log_support)
                    if not np.all(np.diff(new_support) > 0):
                        continue
                    new_val = _compute_P_fast(new_support, weights)
                    if new_val > base_val + 1e-10:
                        base_val = new_val
                        log_support[i] += delta
                        support = np.exp(log_support)
                        improved = True
                        break
            if not improved:
                break
        
        # Evaluate current solution
        cur_val = _compute_P_fast(support, weights)
        if cur_val > best_val:
            best_val = cur_val
            best_support = support.copy()
            best_weights = weights.copy()
        
        # Check for convergence
        if it > 0 and abs(cur_val - best_val) < 1e-12:
            break
    
    return best_val, best_support, best_weights


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    global _START
    _START = time.time()
    
    best_val = -1.0
    best_support = None
    best_weights = None
    
    # Try different numbers of support points
    for n in [3, 4, 5, 6, 8]:
        if not _deadline_ok(margin=120.0):
            break
        
        # Log grid range: from 0.01 to 100
        log_min = np.log(0.01)
        log_max = np.log(100.0)
        
        val, support, weights = _log_grid_search(n, log_min, log_max, max_iter=30)
        
        if val > best_val:
            best_val = val
            best_support = support
            best_weights = weights
    
    # Also try the 3-point family {0, 1, M} which was effective
    if _deadline_ok(margin=60.0):
        M_grid = np.logspace(np.log10(0.2), np.log10(300.0), 50)
        u_grid = np.linspace(0.002, 0.998, 20)
        t_grid = np.linspace(0.002, 0.998, 20)
        
        for M in M_grid:
            if not _deadline_ok(margin=60.0):
                break
            for u in u_grid:
                for t in t_grid:
                    w = np.array([u * t, 1.0 - t, (1.0 - u) * t])
                    s = np.array([0.0, 1.0, M])
                    v = _compute_P_fast(s, w)
                    if v > best_val:
                        best_val = v
                        best_support = s.copy()
                        best_weights = w.copy()
    
    # Fallback
    if best_support is None:
        n = 16
        support = np.logspace(np.log(0.01), np.log(100.0), n)
        weights = np.ones(n) / n
        return support, weights
    
    support = np.asarray(best_support, dtype=np.float64).ravel()
    weights = np.asarray(best_weights, dtype=np.float64).ravel()
    weights = weights / weights.sum()
    return support, weights
# EVOLVE-BLOCK-END
