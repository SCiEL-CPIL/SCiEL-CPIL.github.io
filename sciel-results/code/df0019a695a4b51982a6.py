"""Optimize equal-width step heights to minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _max_overlap(h):
    """Exact max overlap: max_k sum_i h[i]*(1-h[i-k]) for all shifts k."""
    one_minus = 1.0 - h
    best = 0.0
    n = len(h)
    for k in range(n):
        a = h[k:]
        b = one_minus[:n - k]
        ov = np.dot(a, b)
        if ov > best:
            best = ov
    return best


def _sinusoidal_h(A, phi):
    """Generate step heights from sinusoidal ansatz, projected to [0,1] with sum N/2."""
    i = np.arange(N)
    raw = 0.5 + A * np.sin(2 * np.pi * i / N + phi)
    # Project to [0,1] with sum constraint
    return _project(raw, N / 2.0)


def _project(h, target_sum):
    """Project onto {0<=x<=1, sum x = target_sum} via bisection on Lagrange multiplier."""
    lo, hi = -100.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2.0
    return np.clip(h - mid, 0.0, 1.0)


def _overlap_from_sin(A, phi):
    """Compute max overlap for sinusoidal parameterization."""
    h = _sinusoidal_h(A, phi)
    return _max_overlap(h)


def _objective_sin(params):
    """Objective for scipy minimize: returns max overlap for (A, phi)."""
    A, phi = params
    return _overlap_from_sin(A, phi)


def construct():
    target = N / 2.0
    
    # Start with uniform baseline
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Grid search over A and phi
    A_values = np.linspace(0.0, 0.5, 11)  # A in [0, 0.5]
    phi_values = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    
    for A in A_values:
        if time.time() > DEADLINE - 10.0:
            break
        for phi in phi_values:
            if time.time() > DEADLINE - 10.0:
                break
            h = _sinusoidal_h(A, phi)
            score = _max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Local refinement using scipy minimize around best grid point
    # Find best (A, phi) from grid
    best_A, best_phi = 0.0, 0.0
    best_grid_score = best_score
    for A in A_values:
        for phi in phi_values:
            h = _sinusoidal_h(A, phi)
            score = _max_overlap(h)
            if score < best_grid_score:
                best_grid_score = score
                best_A, best_phi = A, phi
    
    # Local Newton refinement
    if time.time() < DEADLINE - 20.0:
        x0 = [best_A, best_phi]
        try:
            result = minimize(_objective_sin, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                h = _sinusoidal_h(result.x[0], result.x[1])
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        except Exception:
            pass
    
    # Also try full-vector L-BFGS-B from the best sinusoidal point
    if time.time() < DEADLINE - 30.0:
        h_start = best_h.copy()
        
        def _full_objective(h):
            return _max_overlap(h)
        
        def _full_grad(h):
            # Gradient of max overlap (use subgradient of the active shift)
            one_minus = 1.0 - h
            n = len(h)
            vals = np.empty(n)
            for k in range(n):
                a = h[k:]
                b = one_minus[:n - k]
                vals[k] = np.dot(a, b)
            k_max = np.argmax(vals)
            
            grad = np.zeros(n)
            # contribution from h[i] for i in [k_max, n-1]: (1-h[i-k_max])
            grad[k_max:] += (1.0 - h[:n-k_max])
            # contribution from (1-h[i]) for i in [0, n-k_max-1]: -h[i+k_max]
            grad[:n-k_max] += (-h[k_max:])
            return grad
        
        try:
            result = minimize(_full_objective, h_start, method='L-BFGS-B', 
                            jac=_full_grad,
                            bounds=[(0, 1)] * N,
                            constraints={'type': 'eq', 'fun': lambda h: np.sum(h) - target,
                                        'jac': lambda h: np.ones(N)},
                            options={'maxiter': 1000, 'ftol': 1e-10})
            if result.fun < best_score:
                h = _project(result.x, target)
                score = _max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = _project(best_h, target)
    return best_h
