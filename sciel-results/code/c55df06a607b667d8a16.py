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


def _fast_metric(h):
    """Vectorized metric using FFT for speed."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0 or l2_sq <= 0:
        return 0.0
    # Full convolution via FFT
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)
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
        v = _fast_metric(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Simple candidates
    h = np.zeros(N); h[0] = 1.0; consider(h)
    h = np.ones(N); consider(h)

    # Top-hats
    for K in range(1, N + 1):
        if time.time() > deadline - 60.0:
            break
        for s in range(0, N - K + 1):
            h = np.zeros(N)
            h[s:s + K] = 1.0
            consider(h)

    # Sparse random candidates - try to find good support sets
    if time.time() < deadline - 60.0:
        np.random.seed(42)
        for trial in range(100):
            if time.time() > deadline - 60.0:
                break
            # Random sparse support
            k = np.random.randint(3, 15)
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            consider(h)
            # Also try with random weights
            for w in np.random.rand(k):
                h2 = np.zeros(N)
                h2[idx] = 0.5 + 0.5 * w
                consider(h2)

    # Differential evolution for global search
    if time.time() < deadline - 45.0:
        bounds = [(0.0, 1.0)] * N
        def neg_metric(x):
            return -_fast_metric(x)
        try:
            res = differential_evolution(neg_metric, bounds, maxiter=50,
                                         tol=1e-6, seed=42, polish=True,
                                         popsize=15)
            consider(res.x)
        except Exception:
            pass

    # Local refinement from best found
    if best_h is not None and time.time() < deadline - 20.0:
        def neg_metric(x):
            return -_fast_metric(np.maximum(x, 0.0))
        try:
            res = minimize(neg_metric, best_h, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            consider(res.x)
        except Exception:
            pass

    # Additional random restarts with local optimization
    if time.time() < deadline - 15.0:
        np.random.seed(123)
        for trial in range(10):
            if time.time() > deadline - 15.0:
                break
            # Sparse initial guess
            k = np.random.randint(3, 12)
            idx = np.random.choice(N, k, replace=False)
            h0 = np.zeros(N)
            h0[idx] = np.random.rand(k)
            def neg_metric2(x):
                return -_fast_metric(np.maximum(x, 0.0))
            try:
                res = minimize(neg_metric2, h0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                consider(res.x)
            except Exception:
                pass

    # Try to exploit sparsity: find optimal support via combinatorial search
    if time.time() < deadline - 10.0:
        # For small support sizes, brute force is feasible
        from itertools import combinations
        for k in range(2, 8):
            if time.time() > deadline - 10.0:
                break
            for combo in combinations(range(N), k):
                h = np.zeros(N)
                for i in combo:
                    h[i] = 1.0
                consider(h)

    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    # Final verification with exact metric
    final_val = _metric(best_h)
    if final_val > 0:
        return list(best_h)
    return list(np.zeros(N) + np.array([1.0] + [0.0]*(N-1)))
