"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Represent the step function as a mixture of Gaussian basis functions
with variable centers, widths, and weights. Optimize these parameters using
differential evolution to exploit smooth continuous structure.
Added: FFT-based autoconvolution for speed, and a final polish pass with
tighter tolerance and more iterations to squeeze out additional improvement.
"""

import time
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600
DELTA = 0.5 / N
DEADLINE = 145.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_fft(h):
    """Compute full autoconvolution via FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * H, size)
    return C[:2 * n - 1]


def _evaluate_step_heights(h):
    """Compute the evaluator's score: max(autoconv) / (sum h)^2."""
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    # Compute autoconvolution via FFT
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    # L1-based normalization: peak / (sum h)^2
    return peak / (s * s)


def _gaussian_mixture_to_heights(params, x_mid):
    """Convert Gaussian mixture parameters to step heights.
    
    params: [w1, c1, s1, w2, c2, s2, ..., wK, cK, sK]
    where K is number of Gaussians.
    """
    K = len(params) // 3
    h = np.zeros_like(x_mid)
    for k in range(K):
        w = params[3*k]
        c = params[3*k+1]
        s = params[3*k+2]
        s = max(s, 1e-6)  # avoid zero width
        h += w * np.exp(-0.5 * ((x_mid - c) / s) ** 2)
    return np.clip(h, 0.0, None)


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Start with uniform as baseline
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_score = _evaluate_step_heights(uniform)
    
    # Try Gaussian mixture with K=1, 2, 3 Gaussians
    for K in [1, 2, 3]:
        if _time_left() < 20.0:
            break
        
        # Bounds for each parameter: weight in [0, 2], center in [-0.25, 0.25], sigma in [0.01, 0.25]
        bounds = [(0.0, 2.0), (-0.25, 0.25), (0.01, 0.25)] * K
        
        def objective(params):
            if _time_left() < 5.0:
                return float('inf')
            h = _gaussian_mixture_to_heights(params, x_mid)
            return _evaluate_step_heights(h)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.success or result.fun < best_score:
                h_candidate = _gaussian_mixture_to_heights(result.x, x_mid)
                score = _evaluate_step_heights(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Also try a simple parametric family: h(x) = 1 + eps*cos(2*pi*x/0.5)
    for eps in np.linspace(-0.5, 0.5, 21):
        if _time_left() < 5.0:
            break
        h = 1.0 + eps * np.cos(2 * np.pi * x_mid / 0.5)
        h = np.clip(h, 0.0, None)
        score = _evaluate_step_heights(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final polish: use scipy.optimize.minimize on the continuous parameters
    # of the best Gaussian mixture found, with tighter tolerance
    if _time_left() > 10.0 and best_h is not uniform:
        # Find the best K from previous search by checking which mixture
        # gave the best result. For simplicity, try K=2 and K=3.
        for K in [2, 3]:
            if _time_left() < 5.0:
                break
            bounds = [(0.0, 2.0), (-0.25, 0.25), (0.01, 0.25)] * K
            
            def objective_polish(params):
                if _time_left() < 2.0:
                    return float('inf')
                h = _gaussian_mixture_to_heights(params, x_mid)
                return _evaluate_step_heights(h)
            
            # Initialize with random points near the current best
            # We don't have the exact params, so use a heuristic init
            x0 = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=len(bounds)
            )
            # Try a few random starts
            for trial in range(3):
                if _time_left() < 3.0:
                    break
                x0_trial = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds],
                    size=len(bounds)
                )
                try:
                    res = minimize(
                        objective_polish,
                        x0_trial,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < best_score:
                        h_candidate = _gaussian_mixture_to_heights(res.x, x_mid)
                        score = _evaluate_step_heights(h_candidate)
                        if score < best_score:
                            best_score = score
                            best_h = h_candidate.copy()
                except Exception:
                    pass
    
    # Ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
