import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _compute_P(support, weights):
    """Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum

    # Pair sums and pair probabilities
    pair_sums = s[:, None] + s[None, :]      # (n, n)
    pair_probs = np.outer(w, w)              # (n, n)

    # threshold[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]  # (n, n)

    # inner_sum[k, l] = sum_{i,j} pair_probs[i,j] * I(pair_sums[i,j] < thresholds[k,l])
    ps = pair_sums[None, None, :, :]   # (1, 1, n, n)
    th = thresholds[:, :, None, None]  # (n, n, 1, 1)
    indicator = (ps < th).astype(np.float64)  # (n, n, n, n)
    pp = pair_probs[None, None, :, :]  # (1, 1, n, n)
    inner_sum = np.sum(pp * indicator, axis=(2, 3))  # (n, n)

    wl = np.outer(w, w)
    P = np.sum(wl * inner_sum)
    return float(P)


def _neg_P(params, n):
    """Objective: negative P for a sparse support {0, 1, G} with n=3.
    params = [logG, w0, w1, w2] (weights not normalized here)."""
    logG = params[0]
    G = float(np.exp(np.clip(logG, 0.0, 8.0)))  # G in [1, e^8 ~ 2981]
    s = np.array([0.0, 1.0, G], dtype=np.float64)
    w = np.maximum(params[1:], 1e-12)
    return -_compute_P(s, w)


def _neg_P_2pt(params):
    """Binary support {0, 1}, params = [w0, w1]."""
    s = np.array([0.0, 1.0], dtype=np.float64)
    w = np.maximum(params, 1e-12)
    return -_compute_P(s, w)


def _optimize_sparse_3pt(deadline):
    """Optimize over the sparse 3-point support {0,1,G}."""
    start = time.time()
    bounds = [(0.0, 8.0), (1e-4, 1.0), (1e-4, 1.0), (1e-4, 1.0)]

    def _time_check():
        return (time.time() - start) < (deadline - 5.0)

    try:
        result = differential_evolution(
            _neg_P, bounds, args=(3,),
            maxiter=60, popsize=12, tol=1e-7,
            seed=123, polish=False, init='sobol',
            update_interval=1, callback=_time_check,
        )
    except Exception:
        return None, None, 0.0

    if result.x is None:
        return None, None, 0.0

    # Polish with Nelder-Mead
    try:
        res2 = minimize(
            _neg_P, result.x, args=(3,),
            method='Nelder-Mead',
            options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True},
        )
        if res2.fun < result.fun:
            result = res2
    except Exception:
        pass

    logG = float(np.clip(result.x[0], 0.0, 8.0))
    G = float(np.exp(logG))
    s = np.array([0.0, 1.0, G], dtype=np.float64)
    w = np.maximum(result.x[1:], 1e-12)
    P = _compute_P(s, w)
    return s, w, P


def _optimize_binary(deadline):
    """Optimize over binary support {0,1}."""
    start = time.time()
    bounds = [(1e-4, 1.0), (1e-4, 1.0)]

    def _time_check():
        return (time.time() - start) < (deadline - 5.0)

    try:
        result = differential_evolution(
            _neg_P_2pt, bounds, args=(),
            maxiter=40, popsize=10, tol=1e-7,
            seed=7, polish=False, init='sobol',
            update_interval=1, callback=_time_check,
        )
    except Exception:
        return None, None, 0.0

    if result.x is None:
        return None, None, 0.0

    s = np.array([0.0, 1.0], dtype=np.float64)
    w = np.maximum(result.x, 1e-12)
    P = _compute_P(s, w)
    return s, w, P


def _analytic_binary():
    """Known analytic optimum for binary {0,1}."""
    p = (1.0 + np.sqrt(33.0)) / 16.0
    s = np.array([0.0, 1.0], dtype=np.float64)
    w = np.array([1.0 - p, p], dtype=np.float64)
    P = _compute_P(s, w)
    return s, w, P


def _sparse3_init(G, w0, w1, w2):
    """Quick analytic-ish initialization for the 3-point support."""
    s = np.array([0.0, 1.0, float(G)], dtype=np.float64)
    w = np.array([w0, w1, w2], dtype=np.float64)
    P = _compute_P(s, w)
    return s, w, P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # --- Baselines (always valid) ---
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    best_P = _compute_P(support_base, weights_base)

    # --- Analytic binary ---
    try:
        s, w, P = _analytic_binary()
        if P > best_P:
            best_P, best_support, best_weights = P, s, w
    except Exception:
        pass

    # --- Sparse 3-point optimization (main structural play) ---
    if time.time() < deadline - 60:
        try:
            s, w, P = _optimize_sparse_3pt(deadline)
            if s is not None and P > best_P:
                best_P, best_support, best_weights = P, s, w
        except Exception:
            pass

    # --- Binary scipy (cheap, may beat analytic slightly) ---
    if time.time() < deadline - 30:
        try:
            s, w, P = _optimize_binary(deadline)
            if s is not None and P > best_P:
                best_P, best_support, best_weights = P, s, w
        except Exception:
            pass

    # --- A few hand-tuned sparse 3-point initializations as safety net ---
    if time.time() < deadline - 20:
        for G, w0, w1, w2 in [
            (10.0, 0.10, 0.10, 0.80),
            (20.0, 0.15, 0.15, 0.70),
            (5.0, 0.20, 0.20, 0.60),
            (30.0, 0.05, 0.10, 0.85),
        ]:
            try:
                s, w, P = _sparse3_init(G, w0, w1, w2)
                if P > best_P:
                    best_P, best_support, best_weights = P, s, w
            except Exception:
                pass

    # --- Final validation / fallback ---
    if best_support is None or best_weights is None:
        return support_base, weights_base

    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return support_base, weights_base

    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 1e-12)

    if np.sum(best_weights) <= 1e-9:
        return support_base, weights_base

    if len(best_support) != len(best_weights) or len(best_support) < 1:
        return support_base, weights_base

    return best_support, best_weights
