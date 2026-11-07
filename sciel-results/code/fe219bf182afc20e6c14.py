"""Rewritten program: Parametric shape search (Gaussian, Polynomial, Cosine)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 125.0  # safety margin well under the 150s kill


def _evaluate(h):
    """Exact task score for nonnegative heights h (length N)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


# Precompute x-coordinates for the 50 points in [-1/4, 1/4]
# np.linspace(-0.25, 0.25, 50)
X = np.linspace(-0.25, 0.25, N)


def _gen_gaussian(alpha):
    """Generate Gaussian bump heights."""
    h = np.exp(-alpha * X ** 2)
    return h


def _gen_polynomial(beta):
    """Generate Polynomial bump heights: (1 - (x/0.25)^2)^beta."""
    x_norm = X / 0.25
    base = np.clip(1.0 - x_norm ** 2, 0.0, 1.0)
    h = base ** beta
    return h


def _gen_cosine(gamma):
    """Generate Cosine bump heights: cos(gamma * x). Ensure non-negative."""
    h = np.cos(gamma * X)
    h = np.clip(h, 0.0, None)
    return h


def construct():
    start = time.time()

    def elapsed():
        return time.time() - start

    best_h = np.ones(N)
    best_s = _evaluate(best_h)

    # Define parametric families and their parameter ranges
    # We optimize the parameter 'p' for each family
    families = [
        (lambda p: _gen_gaussian(p[0]), (0.1, 100.0), "Gaussian"),
        (lambda p: _gen_polynomial(p[0]), (0.5, 20.0), "Polynomial"),
        (lambda p: _gen_cosine(p[0]), (1.0, 20.0), "Cosine"),
    ]

    for gen_func, bounds, name in families:
        if elapsed() > DEADLINE - 5:
            break
            
        # Objective function for minimizer (we maximize score, so minimize -score)
        def objective(p):
            h = gen_func(p)
            if np.all(h == 0):
                return 0.0
            # Normalize to max 1 to avoid scaling issues, though score is scale-invariant?
            # value = l2 / (norm_1 * norm_inf).
            # l2 is quadratic in h. norm_1 is linear in h. norm_inf is linear in h.
            # So value is invariant to scaling of h.
            # Thus we don't strictly need to normalize, but it helps numerical stability.
            mx = np.max(h)
            if mx > 0:
                h = h / mx
            s = _evaluate(h)
            return -s
        
        # Initial guesses
        initial_guesses = [
            np.array([bounds[0] * 1.1]),
            np.array([bounds[1] * 0.9]),
            np.array([(bounds[0] + bounds[1]) / 2.0]),
            np.array([np.sqrt(bounds[0] * bounds[1])]),
        ]
        
        # Remove duplicates
        unique_guesses = []
        for g in initial_guesses:
            if not any(np.allclose(g, ug, atol=1e-3) for ug in unique_guesses):
                unique_guesses.append(g)
        
        for x0 in unique_guesses:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.success or res.fun < 0:
                    best_p = res.x
                    h_candidate = gen_func(best_p)
                    mx = np.max(h_candidate)
                    if mx > 0:
                        h_candidate = h_candidate / mx
                    s_candidate = _evaluate(h_candidate)
                    if s_candidate > best_s:
                        best_s = s_candidate
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Ensure non-negative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h[~np.isfinite(best_h)] = 0.0
    
    # Fallback if all failed (should not happen)
    if np.all(best_h == 0):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
