import numpy as np
import time
import sys
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0  # Safety margin for 150s limit
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv). Returns -inf/0 if degenerate."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)


def _max_conv(h):
    """Returns the maximum value of the autoconvolution."""
    conv = np.convolve(h, h, mode="full")
    return np.max(conv)


def _obj_and_grad(h):
    """
    Objective: Minimize max_autoconv(h) subject to sum(h)=1, h>=0.
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0:
        return 0.0, np.zeros(N)
    
    idx_max = np.argmax(conv)
    # Gradient of (h*h)_k with respect to h_j is 2 * h_{k-j} (with 0 padding)
    # h is length N. conv is length 2N-1.
    # idx_max is in [0, 2N-2].
    # h_{k-j} is non-zero if 0 <= k-j < N.
    grad = np.zeros(N)
    for j in range(N):
        idx = idx_max - j
        if 0 <= idx < N:
            grad[j] = 2.0 * h[idx]
        
    return m, grad


def _minimize_peak(h_init, maxiter=500):
    """
    Local optimization to minimize peak autoconvolution given L1=1 normalization.
    """
    h0 = h_init / np.sum(h_init)
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}]
    
    try:
        res = minimize(_obj_and_grad, h0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
        if res.x is not None:
            hx = np.clip(res.x, 0.0, None)
            if np.sum(hx) > 0:
                hx = hx / np.sum(hx)
                return hx
    except Exception:
        pass
    return h0


def _generate_sidon_starts(k, num_seeds=5):
    """Generate initial vectors based on Sidon sets (Golomb rulers)."""
    starts = []
    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)
        S = []
        sums = set()
        candidates = list(range(N))
        rng.shuffle(candidates)
        
        for cand in candidates:
            if len(S) == k:
                break
            conflict = False
            new_sums = set()
            for s in S:
                ps = cand + s
                if ps in sums:
                    conflict = True
                    break
                new_sums.add(ps)
            if not conflict:
                if cand + cand in sums:
                    conflict = True
                else:
                    new_sums.add(cand + cand)
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) >= 2:
            h = np.zeros(N)
            for idx in S:
                h[idx] = 1.0
            starts.append(h)
    return starts


def construct():
    np.random.seed(42)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    # Phase 1: Sidon Set Based Starts
    # Sparse sets often minimize peak autocorrelation relative to L1.
    for k in range(3, 15):
        if _time_left() < 5:
            break
        starts = _generate_sidon_starts(k, num_seeds=10)
        for h_s in starts:
            if _time_left() < 2:
                break
            h_opt = _minimize_peak(h_s, maxiter=300)
            upd(h_opt)

    # Phase 2: Random Smooth Starts
    # Gaussian bumps and other smooth functions
    for _ in range(20):
        if _time_left() < 5:
            break
        a = np.random.uniform(1.0, 10.0)
        c = np.random.uniform(5, N - 5)
        h = np.exp(-a * (np.arange(N) - c) ** 2)
        # Add some noise to break symmetry
        h += 0.1 * np.random.rand(N)
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            h_opt = _minimize_peak(h, maxiter=300)
            upd(h_opt)

    # Phase 3: Random Sparse Starts
    for _ in range(20):
        if _time_left() < 5:
            break
        k = np.random.randint(3, 12)
        h = np.zeros(N)
        h[np.random.choice(N, k, replace=False)] = 1.0
        # Randomize weights slightly
        h *= np.random.uniform(0.5, 1.5, N)
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            h_opt = _minimize_peak(h, maxiter=300)
            upd(h_opt)

    # Phase 4: Polishing the best found
    if _time_left() > 10:
        h_opt = _minimize_peak(best_h, maxiter=1000)
        upd(h_opt)
        
    # Phase 5: Final check and return
    # Ensure we return the best one found
    # The ratio is scale invariant, so we can return any scale, but usually normalized or raw.
    # The task says "sequence of nonnegative step heights".
    # The evaluator computes the ratio, so scale doesn't matter for the score, 
    # but we should return a valid vector.
    return list(best_h)

if __name__ == "__main__":
    # Simple self-test
    h = construct()
    print(f"Score: {_ratio(np.array(h))}")
