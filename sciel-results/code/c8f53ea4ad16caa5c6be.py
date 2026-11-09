# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation).
    Constructs a distribution P[y, x] on a 10x10 grid to maximize
    H(x-y) / max(H(x), H(x+y), H(x+2y)).
    Uses scipy optimization with multiple random restarts.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    n = 10

    # Precompute indices for vectorized computation
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y

    # Precompute the index arrays for each projection
    XY_diff = (X - Y + (n - 1)).flatten()  # 0 to 2*(n-1)
    X_flat = X.flatten()  # 0 to n-1
    XY_sum = (X + Y).flatten()  # 0 to 2*(n-1)
    X2Y = (X + 2 * Y).flatten()  # 0 to 3*(n-1)

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P (vectorized)."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 0.0
        return H_XY / M

    def objective(params):
        """Negative of the ratio for minimization."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    # Start with the current best solution (uniform X-Y)
    P_best = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_best[y, x] = 1.0 / (19.0 * (n - k))
    P_best /= P_best.sum()

    best_ratio = compute_ratio(P_best)

    # Try optimization from multiple starting points
    np.random.seed(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break

        # Random initial point
        P_init = np.random.dirichlet(np.ones(n * n))
        P_init = P_init.reshape(n, n)
        P_init_safe = np.maximum(P_init, 1e-10)
        params_init = np.log(P_init_safe.flatten())

        try:
            result = minimize(objective, params_init, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Also try from the current best with more iterations
    P_best_safe = np.maximum(P_best, 1e-10)
    params_best = np.log(P_best_safe.flatten())
    try:
        result = minimize(objective, params_best, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < -best_ratio:
            P_opt = np.exp(result.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            P_opt /= P_opt.sum()
            ratio = compute_ratio(P_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                P_best = P_opt.copy()
    except Exception:
        pass

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    P_best /= P_best.sum()

    return P_best

# EVOLVE-BLOCK-END
