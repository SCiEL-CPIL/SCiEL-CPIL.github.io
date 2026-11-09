# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline guard: evaluator kills at 630s, we target finishing by ~600s
_DEADLINE = 630.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _shannon_bits(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_entropies(P):
    """
    Given P[y, x] (10x10), compute:
      H_{-1}: entropy of distribution of x - y
      H_0: entropy of distribution of x
      H_1: entropy of distribution of x + y
      H_2: entropy of distribution of x + 2y
    Returns (H_minus1, H_0, H_1, H_2).
    """
    n = P.shape[0]
    ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

    # x - y
    d_minus1 = xs - ys
    vals = np.unique(d_minus1)
    p_minus1 = np.array([P[d_minus1 == v].sum() for v in vals])
    H_minus1 = _shannon_bits(p_minus1)

    # x
    vals_x = np.unique(xs)
    p_x = np.array([P[xs == v].sum() for v in vals_x])
    H_0 = _shannon_bits(p_x)

    # x + y
    d_1 = xs + ys
    vals_1 = np.unique(d_1)
    p_1 = np.array([P[d_1 == v].sum() for v in vals_1])
    H_1 = _shannon_bits(p_1)

    # x + 2y
    d_2 = xs + 2 * ys
    vals_2 = np.unique(d_2)
    p_2 = np.array([P[d_2 == v].sum() for v in vals_2])
    H_2 = _shannon_bits(p_2)

    return H_minus1, H_0, H_1, H_2


def _objective(params):
    """
    Objective to MINIMIZE: -D/M where D = H_{-1}, M = max(H_0, H_1, H_2).
    params: flattened 10x10 array of non-negative weights.
    """
    P = np.clip(params.reshape(10, 10), 0, None)
    s = P.sum()
    if s <= 1e-9:
        return 1e6
    P = P / s

    H_m1, H0, H1, H2 = _compute_entropies(P)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 1e6
    return -(H_m1 / M)


def _make_starting_points(n_starts=20):
    """Generate diverse starting points."""
    starts = []
    n = 10

    # Uniform
    starts.append(np.ones((n, n)) / (n * n))

    # Diagonal emphasis (x - y constant bands)
    for alpha in [0.5, 1.0, 2.0]:
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        d = xs - ys
        P = np.exp(-alpha * d**2 / n)
        P /= P.sum()
        starts.append(P.copy())

    # Gaussian blobs at various centers
    rng = np.random.RandomState(42)
    for _ in range(10):
        cy, cx = rng.uniform(2, 8, 2)
        sigma = rng.uniform(1.0, 3.0)
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        P = np.exp(-((ys - cy)**2 + (xs - cx)**2) / (2 * sigma**2))
        P /= P.sum()
        starts.append(P.copy())

    # Anti-diagonal emphasis (to make x+y or x+2y concentrated)
    for beta in [0.5, 1.0, 1.5]:
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        s = xs + ys
        P = np.exp(-beta * (s - n)**2 / n)
        P /= P.sum()
        starts.append(P.copy())

    # Corner concentration
    for corner in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
        cy, cx = corner
        sigma = 1.5
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        P = np.exp(-((ys - cy)**2 + (xs - cx)**2) / (2 * sigma**2))
        P /= P.sum()
        starts.append(P.copy())

    return starts


def _local_refine(P0, max_iter=200):
    """Use scipy minimize to refine a starting point."""
    x0 = P0.flatten().copy()
    res = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )
    P_best = np.clip(res.x.reshape(10, 10), 0, None)
    s = P_best.sum()
    if s > 1e-9:
        P_best /= s
    else:
        P_best = P0
    return P_best, _objective(P_best.flatten())


def _differential_evolution_search(popsize=15, maxiter=30, tol=1e-8):
    """Use differential evolution for global search."""
    bounds = [(0, 1)] * (10 * 10)
    res = differential_evolution(
        _objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        seed=12345,
        polish=True,
        init='latinhypercube',
        updating='immediate',
        workers=1,
        callback=lambda xk, convergence: convergence
    )
    P_best = np.clip(res.x.reshape(10, 10), 0, None)
    s = P_best.sum()
    if s > 1e-9:
        P_best /= s
    else:
        P_best = np.ones((10, 10)) / 100.0
    return P_best, _objective(P_best.flatten())


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    global _START_TIME
    _START_TIME = time.time()

    best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    best_val = _objective(best_P.flatten())

    def _update_best(P, val):
        nonlocal best_P, best_val
        if val < best_val:  # We're minimizing negative ratio
            best_P = P.copy()
            best_val = val

    # Phase 1: Local refinement from diverse starting points
    starts = _make_starting_points()
    for i, P0 in enumerate(starts):
        if _remaining() < 30:
            break
        try:
            P_refined, val = _local_refine(P0, max_iter=150)
            _update_best(P_refined, val)
        except Exception:
            pass

    # Phase 2: Differential evolution for global exploration
    if _remaining() > 60:
        try:
            P_de, val_de = _differential_evolution_search(popsize=15, maxiter=25)
            _update_best(P_de, val_de)
        except Exception:
            pass

    # Phase 3: Further local refinement from the best found so far
    if _remaining() > 30:
        try:
            P_final, val_final = _local_refine(best_P, max_iter=500)
            _update_best(P_final, val_final)
        except Exception:
            pass

    # Phase 4: One more DE run with different seed if time permits
    if _remaining() > 45:
        try:
            bounds = [(0, 1)] * (10 * 10)
            res = differential_evolution(
                _objective,
                bounds,
                popsize=12,
                maxiter=20,
                tol=1e-8,
                seed=99999,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1
            )
            P_de2 = np.clip(res.x.reshape(10, 10), 0, None)
            s = P_de2.sum()
            if s > 1e-9:
                P_de2 /= s
                val_de2 = _objective(P_de2.flatten())
                _update_best(P_de2, val_de2)
        except Exception:
            pass

    # Final local polish
    if _remaining() > 15:
        try:
            P_final2, val_final2 = _local_refine(best_P, max_iter=300)
            _update_best(P_final2, val_final2)
        except Exception:
            pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    else:
        best_P /= s

    # Final sanity check
    best_P = np.nan_to_num(best_P, nan=0.0, posinf=0.0, neginf=0.0)
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0
    else:
        best_P /= s

    return best_P


# EVOLVE-BLOCK-END
