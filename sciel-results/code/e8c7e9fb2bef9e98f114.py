import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50


def _metric(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) as stated in the task."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -1.0
    h = np.maximum(h, 0.0)
    l2_sq = float(np.sum(h**2))
    l1 = float(np.sum(h))
    if l1 <= 0 or l2_sq <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup = float(np.max(conv))
    if sup <= 0:
        return 0.0
    return l2_sq / (l1 * sup)


def construct():
    start = time.time()
    deadline = start + 145.0

    best_h = None
    best_val = 0.0

    def consider(h):
        nonlocal best_h, best_val
        h = np.maximum(h, 0.0)
        v = _metric(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Simple candidates
    h = np.zeros(N); h[0] = 1.0; consider(h)
    h = np.ones(N); consider(h)

    # Top-hats
    for K in range(1, N + 1):
        if time.time() > deadline - 40.0:
            break
        for s in range(0, N - K + 1):
            h = np.zeros(N)
            h[s:s + K] = 1.0
            consider(h)

    # Differential evolution for global search
    if time.time() < deadline - 30.0:
        bounds = [(0.0, 1.0)] * N
        def neg_metric(x):
            return -_metric(x)
        try:
            res = differential_evolution(neg_metric, bounds, maxiter=40,
                                         tol=1e-6, seed=42, polish=True,
                                         popsize=15)
            consider(res.x)
        except Exception:
            pass

    # Local refinement from best found
    if best_h is not None and time.time() < deadline - 15.0:
        def neg_metric(x):
            return -_metric(np.maximum(x, 0.0))
        try:
            res = minimize(neg_metric, best_h, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            consider(res.x)
        except Exception:
            pass

    # Additional random restarts with local optimization
    if time.time() < deadline - 10.0:
        np.random.seed(123)
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            h0 = np.random.rand(N)
            def neg_metric2(x):
                return -_metric(np.maximum(x, 0.0))
            try:
                res = minimize(neg_metric2, h0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
                consider(res.x)
            except Exception:
                pass

    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
