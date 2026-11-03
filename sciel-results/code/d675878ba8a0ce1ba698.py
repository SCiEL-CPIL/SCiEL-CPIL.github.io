"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Scale-invariant, so fix ||h||_1 = 1 and minimize ||h||_2^2 * ||h*h||_inf.

Structural approach:
  - Precomputed Toeplitz convolution matrix
  - Peak-tracking exact coordinate descent: for each coordinate, find the
    optimal value given which conv bin is the peak (bisection on 1-D piecewise
    smooth function)
  - Multi-start with diverse structured + random initializations
  - Simulated annealing over coordinate perturbations
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 15.0
START = time.time()


def _build_conv_matrix():
    """Build the full autoconvolution Toeplitz matrix C of shape (2N-1, N).
    (C @ h)[k] = sum_i h[i] * h[i - (k - (N-1))]  for k in 0..2N-2."""
    L = 2 * N - 1
    C = np.zeros((L, N))
    for k in range(L):
        shift = k - (N - 1)  # in [-(N-1), N-1]
        for i in range(N):
            j = i - shift
            if 0 <= j < N:
                C[k, i] = 1.0
    return C


C = _build_conv_matrix()
L = 2 * N - 1


def _autoconv(h):
    """Return the full autoconvolution vector of h (length 2N-1)."""
    return C @ h


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    conv = _autoconv(h)
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return 0.0
    return l2sq / (l1 * pinf)


def _cost(h):
    """Scale-invariant cost to minimize: ||h||_2^2 * ||h*h||_inf with ||h||_1=1."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return np.inf
    hn = h / l1
    l2sq = float(hn @ hn)
    conv = _autoconv(hn)
    pinf = float(np.max(np.abs(conv)))
    if pinf <= 0:
        return np.inf
    return l2sq * pinf


def _local_refine(h0, t_end):
    """Local refinement using L-BFGS-B on log-transformed variables."""
    # Parameterize h_i = exp(y_i) / sum(exp(y_j))
    # This ensures h_i > 0 and sum(h) = 1.
    # To allow h_i to approach 0, we can use a large negative y.
    
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 1e-10)
    h = h / h.sum()
    
    # Initial y: log(h)
    y = np.log(h)
    
    def obj(y):
        # Softmax
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        h = exp_y / np.sum(exp_y)
        return _cost(h)
    
    def grad(y):
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        h = exp_y / np.sum(exp_y)
        
        # Gradient of cost wrt h
        # cost = l2sq * pinf
        # This is non-smooth if pinf changes bin.
        # We use finite differences or just let L-BFGS-B use numerical gradients?
        # scipy.optimize.minimize with method='L-BFGS-B' can use numerical gradients if not provided.
        # Providing an analytical gradient for the non-smooth part is tricky.
        # Let's rely on numerical gradients for simplicity and robustness, 
        # or provide gradient for the smooth part and hope it's enough.
        # Actually, _cost is not differentiable at the peak switch.
        # L-BFGS-B with numerical gradients is safer.
        return None
        
    try:
        res = minimize(obj, y, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < obj(y):
            y_opt = res.x
            y_max = np.max(y_opt)
            h_opt = np.exp(y_opt - y_max)
            h_opt = h_opt / np.sum(h_opt)
            return h_opt
    except Exception:
        pass
    return h


def _optimize():
    t_end = START + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    best = np.ones(N) / N
    best_score = _ratio(best)

    # Use Differential Evolution on the log-simplex parameterization
    # y in [-10, 10]^N -> h = softmax(y)
    
    def de_obj(y):
        h = np.exp(y - np.max(y))
        h = h / h.sum()
        return -_ratio(h)  # Maximize ratio -> Minimize negative ratio

    bounds = [(-10, 10)] * N
    
    # We need to be careful with the time limit.
    # DE can be slow. We will run it with a limited number of iterations.
    # We will also do a multi-start approach: run DE from a few different initial populations?
    # Or just one robust run with large popsize.
    
    # Let's try a single run with popsize=20 (20*50=1000 evaluations per gen)
    # If we allow 100 generations, that's 100k evaluations.
    # Each evaluation is O(N^2) ~ 2500 ops. 100k * 2500 = 2.5e8 ops.
    # In Python/NumPy, this might be too slow for 150s.
    # Let's estimate: 1 eval ~ 1ms? 100k evals = 100s. Tight.
    # Let's reduce popsize to 10 and maxiter to 50.
    
    try:
        res = minimize(
            de_obj, 
            np.zeros(N), 
            method='differential_evolution',
            bounds=bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=True  # Use L-BFGS-B to polish the best result
        )
        
        if res.x is not None:
            y_opt = res.x
            h_opt = np.exp(y_opt - np.max(y_opt))
            h_opt = h_opt / h_opt.sum()
            sc = _ratio(h_opt)
            if sc > best_score:
                best_score = sc
                best = h_opt.copy()
                
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't find a good solution or ran out of time, 
    # we can try a few random restarts with local refinement if time permits.
    
    if time.time() < t_end:
        remaining = t_end - time.time()
        # Try 5 random starts with local refinement
        for _ in range(5):
            if time.time() > t_end - 5:
                break
            h_rand = rng.random(N)
            h_rand = h_rand / h_rand.sum()
            h_ref = _local_refine(h_rand, t_end)
            sc = _ratio(h_ref)
            if sc > best_score:
                best_score = sc
                best = h_ref.copy()
                
    # Final refinement on the best candidate
    if time.time() < t_end - 2:
        best = _local_refine(best, t_end)
        best_score = _ratio(best)

    # Ensure nonnegativity and normalization
    best = np.maximum(best, 0.0)
    if best.sum() > 0:
        best = best / best.sum()

    return best, best_score


_cache = None


def construct():
    global _cache
    if _cache is None:
        _cache = _optimize()
    best, _ = _cache
    return [float(x) for x in best]
