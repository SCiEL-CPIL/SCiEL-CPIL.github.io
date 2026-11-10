# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_BUDGET = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_exceeded():
    return (time.time() - _START_TIME) >= (_TIME_BUDGET - _SAFETY_MARGIN)


def _entropy_of_counts(counts):
    """Compute Shannon entropy in bits from raw counts, ignoring zero counts."""
    nz = counts > 0
    if not np.any(nz):
        return 0.0
    p = counts[nz] / counts[nz].sum()
    return -np.sum(p * np.log2(p))


def _compute_ratio(P):
    """
    Compute D/M ratio where D = H_{-1} (x-y) and M = max(H_0, H_1, H_2).
    P is a 10x10 array with P[y, x] = probability.
    Returns (D, M, ratio) or None if M <= 1e-9.
    """
    n = 10
    # P[y, x], y is row, x is column
    # For slope r, value = x + r*y
    # r = -1: value = x - y
    # r = 0: value = x
    # r = 1: value = x + y
    # r = 2: value = x + 2*y

    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)

    # x - y: range from -9 to 9
    diff = xs[None, :] - ys[:, None]  # shape (n, n), diff[y, x] = x - y
    # r=0: x, range 0..9
    x_val = xs[None, :]
    # r=1: x + y, range 0..18
    sum_val = xs[None, :] + ys[:, None]
    # r=2: x + 2y, range 0..27
    twoy_val = xs[None, :] + 2 * ys[:, None]

    def pooled_entropy(matrix, P):
        # P has same shape as matrix
        # Pool by value
        vals = matrix.astype(int)
        min_v = vals.min()
        max_v = vals.max()
        counts = np.zeros(max_v - min_v + 1, dtype=np.float64)
        # Vectorized: for each unique value, sum P where matrix == value
        for v in range(min_v, max_v + 1):
            mask = (vals == v)
            if np.any(mask):
                counts[v - min_v] = P[mask].sum()
        return _entropy_of_counts(counts)

    D = pooled_entropy(diff, P)
    H0 = pooled_entropy(x_val, P)
    H1 = pooled_entropy(sum_val, P)
    H2 = pooled_entropy(twoy_val, P)
    M = max(H0, H1, H2)

    if M <= 1e-9:
        return None
    return D, M, D / M


def _compute_ratio_vectorized(P):
    """
    Faster computation using np.add.at or bincount-style pooling.
    """
    n = 10
    xs = np.arange(n, dtype=np.int32)
    ys = np.arange(n, dtype=np.int32)

    # diff: x - y, offset by n-1 to make non-negative
    diff_idx = xs[None, :] - ys[:, None] + (n - 1)  # range 0..18
    x_idx = xs[None, :]  # range 0..9
    sum_idx = xs[None, :] + ys[:, None]  # range 0..18
    twoy_idx = xs[None, :] + 2 * ys[:, None]  # range 0..27

    # Flatten P
    Pf = P.ravel()

    def pool_entropy(idx_flat, P_flat, num_bins):
        counts = np.zeros(num_bins, dtype=np.float64)
        np.add.at(counts, idx_flat, P_flat)
        return _entropy_of_counts(counts)

    D = pool_entropy(diff_idx.ravel(), Pf, 2 * n - 1)
    H0 = pool_entropy(x_idx.ravel(), Pf, n)
    H1 = pool_entropy(sum_idx.ravel(), Pf, 2 * n - 1)
    H2 = pool_entropy(twoy_idx.ravel(), Pf, 3 * n - 1)
    M = max(H0, H1, H2)

    if M <= 1e-9:
        return None
    return D, M, D / M


def _soft_max_entropy(P, beta=1.0):
    """
    Compute soft-max version of max(H0, H1, H2) using log-sum-exp.
    This provides a smooth surrogate for the max in M.
    """
    n = 10
    xs = np.arange(n, dtype=np.int32)
    ys = np.arange(n, dtype=np.int32)

    diff_idx = xs[None, :] - ys[:, None] + (n - 1)
    x_idx = xs[None, :]
    sum_idx = xs[None, :] + ys[:, None]
    twoy_idx = xs[None, :] + 2 * ys[:, None]

    Pf = P.ravel()

    def pool_entropy(idx_flat, P_flat, num_bins):
        counts = np.zeros(num_bins, dtype=np.float64)
        np.add.at(counts, idx_flat, P_flat)
        return _entropy_of_counts(counts)

    D = pool_entropy(diff_idx.ravel(), Pf, 2 * n - 1)
    H0 = pool_entropy(x_idx.ravel(), Pf, n)
    H1 = pool_entropy(sum_idx.ravel(), Pf, 2 * n - 1)
    H2 = pool_entropy(twoy_idx.ravel(), Pf, 3 * n - 1)

    # Soft max of H0, H1, H2
    Hs = np.array([H0, H1, H2])
    max_H = Hs.max()
    if max_H < 1e-15:
        return D, 0.0, 0.0, None
    log_sum_exp = max_H + np.log(np.sum(np.exp(Hs - max_H)))
    M_soft = log_sum_exp / np.log(2.0)  # In bits? Actually log_sum_exp is in nats if Hs are in bits... 
    # Wait, Hs are in bits. Let me redo.
    # Actually, let's just use the hard max for the true ratio, and use soft for optimization.
    # For now, return true values.
    M = max_H
    if M <= 1e-9:
        return D, M, 0.0, None
    return D, M, D / M, (H0, H1, H2)


def _gradient_ratio(P, eps=1e-7):
    """
    Numerical gradient of the ratio D/M with respect to P (before normalization).
    We'll use central differences on the log-ratio for stability.
    Actually, let's compute gradient of log(D) - log(M) with respect to P.
    """
    n = 10
    N = n * n
    grad = np.zeros(N, dtype=np.float64)
    P_flat = P.ravel().copy()
    
    # We need to be careful: P should sum to 1 and be non-negative.
    # For gradient computation, we'll perturb P_flat and renormalize.
    # Actually, let's work in log-space or use the simplex constraint.
    # Simpler: compute ratio at P + eps*e_i (renormalized) for each i.
    
    # First, compute base ratio
    base = _compute_ratio_vectorized(P)
    if base is None:
        return grad
    D0, M0, ratio0 = base
    
    # For each cell, perturb
    for i in range(N):
        P_plus = P_flat.copy()
        P_plus[i] += eps
        P_plus = np.maximum(P_plus, 0)
        P_plus /= P_plus.sum()
        
        P_minus = P_flat.copy()
        P_minus[i] -= eps
        P_minus = np.maximum(P_minus, 0)
        P_minus /= P_minus.sum()
        
        res_plus = _compute_ratio_vectorized(P_plus.reshape(n, n))
        res_minus = _compute_ratio_vectorized(P_minus.reshape(n, n))
        
        if res_plus is not None and res_minus is not None:
            grad[i] = (res_plus[2] - res_minus[2]) / (2 * eps)
        elif res_plus is not None:
            grad[i] = (res_plus[2] - ratio0) / eps
        elif res_minus is not None:
            grad[i] = (ratio0 - res_minus[2]) / eps
    
    return grad


def _optimize_from_start(P_start, max_iter=200, lr=0.01):
    """
    Simple projected gradient ascent on the simplex for the ratio D/M.
    Uses a smooth surrogate: maximize log(D) - log(M) to handle the ratio smoothly.
    """
    n = 10
    N = n * n
    P = P_start.ravel().copy()
    P = np.maximum(P, 1e-15)
    P /= P.sum()
    
    best_P = P.copy()
    best_val = 0.0
    base = _compute_ratio_vectorized(P.reshape(n, n))
    if base is not None:
        best_val = base[2]
    
    for it in range(max_iter):
        if _deadline_exceeded():
            break
        
        # Compute gradient of log(D) - log(M)
        # Numerical gradient in log-space might be better
        # Let's use the ratio gradient for simplicity
        grad = _gradient_ratio(P.reshape(n, n), eps=1e-6)
        
        # Projected gradient step: P_new = P + lr * grad, then project to simplex
        P_new = P + lr * grad
        P_new = np.maximum(P_new, 0)
        
        # Project to simplex (sum to 1)
        P_new /= P_new.sum()
        
        # Ensure minimum
        P_new = np.maximum(P_new, 1e-15)
        P_new /= P_new.sum()
        
        val = _compute_ratio_vectorized(P_new.reshape(n, n))
        if val is not None and val[2] > best_val:
            best_val = val[2]
            best_P = P_new.copy()
        
        P = P_new
    
    return best_P.reshape(n, n), best_val


def _scipy_optimize(P_start, max_iter=500):
    """
    Use scipy.optimize.minimize with L-BFGS-B on the log-ratio objective.
    Parameterize P via log-softmax to ensure P > 0 and sum(P) = 1.
    """
    from scipy.optimize import minimize
    
    n = 10
    N = n * n
    
    # Convert P_start to log-parameters (z) where P = softmax(z)
    # z has N-1 degrees of freedom (one constraint)
    P_start = P_start.ravel()
    P_start = np.maximum(P_start, 1e-15)
    P_start /= P_start.sum()
    z_start = np.log(P_start) - np.log(P_start).max()
    # Fix last element to 0 for parameterization
    z_free_start = z_start[:-1].copy()
    
    def objective_and_grad(z_free):
        z = np.empty(N)
        z[:-1] = z_free
        z[-1] = 0.0
        # Softmax
        z_max = z.max()
        exp_z = np.exp(z - z_max)
        P = exp_z / exp_z.sum()
        
        res = _compute_ratio_vectorized(P.reshape(n, n))
        if res is None:
            return 1e10, np.zeros(N - 1)
        
        D, M, ratio = res
        # Maximize ratio => minimize -log(ratio) = -log(D) + log(M)
        obj = -np.log(D + 1e-15) + np.log(M + 1e-15)
        
        # Numerical gradient
        grad_free = np.zeros(N - 1)
        eps = 1e-6
        for i in range(N - 1):
            z_plus = z_free.copy()
            z_plus[i] += eps
            z_m = np.empty(N)
            z_m[:-1] = z_plus
            z_m[-1] = 0.0
            z_max_p = z_m.max()
            exp_p = np.exp(z_m - z_max_p)
            P_plus = exp_p / exp_p.sum()
            res_p = _compute_ratio_vectorized(P_plus.reshape(n, n))
            
            z_minus = z_free.copy()
            z_minus[i] -= eps
            z_m2 = np.empty(N)
            z_m2[:-1] = z_minus
            z_m2[-1] = 0.0
            z_max_m = z_m2.max()
            exp_m = np.exp(z_m2 - z_max_m)
            P_minus = exp_m / exp_m.sum()
            res_m = _compute_ratio_vectorized(P_minus.reshape(n, n))
            
            if res_p is not None and res_m is not None:
                obj_p = -np.log(res_p[0] + 1e-15) + np.log(res_p[1] + 1e-15)
                obj_m = -np.log(res_m[0] + 1e-15) + np.log(res_m[1] + 1e-15)
                grad_free[i] = (obj_p - obj_m) / (2 * eps)
        
        return obj, grad_free
    
    result = minimize(
        objective_and_grad, z_free_start, jac=True,
        method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
    )
    
    z = np.empty(N)
    z[:-1] = result.x
    z[-1] = 0.0
    z_max = z.max()
    exp_z = np.exp(z - z_max)
    P = exp_z / exp_z.sum()
    P = P.reshape(n, n)
    
    res = _compute_ratio_vectorized(P)
    val = res[2] if res is not None else 0.0
    
    return P, val


def _structured_starts(n=10):
    """
    Generate structured initial distributions to seed the optimization.
    """
    starts = []
    
    # Uniform
    P = np.ones((n, n)) / (n * n)
    starts.append(P)
    
    # Diagonal concentration (x - y = 0)
    for d in range(-4, 5):
        P = np.zeros((n, n))
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                P[y, x] = 1.0
        P /= P.sum()
        starts.append(P)
    
    # Row concentration (x varies, y fixed) - maximizes H_0
    for y0 in range(n):
        P = np.zeros((n, n))
        P[y0, :] = 1.0
        P /= P.sum()
        starts.append(P)
    
    # Column concentration (y varies, x fixed) - minimizes H_0, H_1, H_2
    for x0 in range(n):
        P = np.zeros((n, n))
        P[:, x0] = 1.0
        P /= P.sum()
        starts.append(P)
    
    # Random sparse starts
    rng = np.random.default_rng(42)
    for trial in range(30):
        P = np.zeros((n, n))
        # Pick 5-15 random cells
        num_cells = rng.integers(5, 16)
        cells = rng.choice(n * n, size=num_cells, replace=False)
        weights = rng.dirichlet(np.ones(num_cells))
        for c, w in zip(cells, weights):
            P[c // n, c % n] = w
        starts.append(P)
    
    # Anti-diagonal (x + y = constant) - this minimizes H_1
    for s in range(0, 2 * n - 1, 2):
        P = np.zeros((n, n))
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            starts.append(P)
    
    return starts


def _polish(P, max_iter=100, lr=0.005):
    """
    Fine-tuning with smaller learning rate and more careful projection.
    """
    n = 10
    P = P.ravel().copy()
    P = np.maximum(P, 1e-15)
    P /= P.sum()
    
    best_P = P.copy()
    base = _compute_ratio_vectorized(P.reshape(n, n))
    best_val = base[2] if base is not None else 0.0
    
    for it in range(max_iter):
        if _deadline_exceeded():
            break
        
        grad = _gradient_ratio(P.reshape(n, n), eps=1e-7)
        P_new = P + lr * grad
        P_new = np.maximum(P_new, 0)
        P_new /= P_new.sum()
        
        val = _compute_ratio_vectorized(P_new.reshape(n, n))
        if val is not None and val[2] > best_val:
            best_val = val[2]
            best_P = P_new.copy()
        
        P = P_new
        # Decay learning rate
        lr *= 0.98
    
    return best_P.reshape(n, n), best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on a 10x10 grid
    so as to maximize the entropy projection ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    n = 10
    
    # Start with uniform
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_val = 0.0
    base = _compute_ratio_vectorized(best_P)
    if base is not None:
        best_val = base[2]
    
    # Try structured starts
    starts = _structured_starts(n)
    
    for P_start in starts:
        if _deadline_exceeded():
            break
        
        # Evaluate start
        res = _compute_ratio_vectorized(P_start)
        if res is not None and res[2] > best_val:
            best_val = res[2]
            best_P = P_start.copy()
        
        # Try scipy optimization from this start
        try:
            P_opt, val_opt = _scipy_optimize(P_start, max_iter=300)
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
        except Exception:
            pass
        
        if _deadline_exceeded():
            break
    
    # Polish from the best found
    try:
        P_polished, val_polished = _polish(best_P, max_iter=200, lr=0.003)
        if val_polished > best_val:
            best_val = val_polished
            best_P = P_polished.copy()
    except Exception:
        pass
    
    # Additional scipy polish from the best
    if not _deadline_exceeded():
        try:
            P_opt2, val_opt2 = _scipy_optimize(best_P, max_iter=400)
            if val_opt2 > best_val:
                best_val = val_opt2
                best_P = P_opt2.copy()
        except Exception:
            pass
    
    # Final safety: ensure validity
    best_P = np.maximum(best_P, 0.0)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= total
    
    # Ensure finite
    best_P = np.nan_to_num(best_P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_P


# EVOLVE-BLOCK-END
