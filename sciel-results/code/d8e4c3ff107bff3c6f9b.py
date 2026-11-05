"""Optimized program using Chebyshev basis and differential evolution."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 400

def _value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _chebyshev_basis(x, max_degree):
    """Compute Chebyshev basis values at points x."""
    basis = np.zeros((max_degree + 1, len(x)))
    for k in range(max_degree + 1):
        if k == 0:
            basis[k] = 1.0
        elif k == 1:
            basis[k] = x
        else:
            basis[k] = 2 * x * basis[k-1] - basis[k-2]
    return basis

def _build_h(coeffs, x, max_degree):
    """Build h from Chebyshev coefficients."""
    basis = _chebyshev_basis(x, max_degree)
    h = basis @ coeffs
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Chebyshev nodes on [-1, 1]
    x = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    x = -x  # Reverse to go from -1 to 1
    
    max_degree = 8
    dim = max_degree + 1
    
    def objective(coeffs):
        h = _build_h(coeffs, x, max_degree)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s ** 2))
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Bounds for coefficients - allow reasonable range
    bounds = [(-5, 5)] * dim
    
    # Run differential evolution with time check
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        if time.time() < deadline - 5.0:
            h_candidate = _build_h(result.x, x, max_degree)
            val_candidate = _value(h_candidate)
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    np.random.seed(123)
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        try:
            init = np.random.uniform(-3, 3, size=dim)
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=12,
                tol=1e-7,
                seed=restart,
                x0=init,
                polish=True
            )
            if time.time() > deadline - 10.0:
                break
            h_candidate = _build_h(result.x, x, max_degree)
            val_candidate = _value(h_candidate)
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
        except Exception:
            continue
    
    # Final check: ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
