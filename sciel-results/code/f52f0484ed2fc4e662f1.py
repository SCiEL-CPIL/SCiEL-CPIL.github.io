"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_ratio(x):
    """Compute ||x||^2 / (||x||_1 * ||x*x||_inf) for nonnegative x."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        return 0.0
    l2_sq = np.sum(x ** 2)
    l1 = np.sum(x)
    if l1 < 1e-30 or l2_sq < 1e-30:
        return 0.0
    # Autoconvolution
    autoconv = np.convolve(x, x, mode='full')
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _negative_ratio(x):
    return -_compute_ratio(x)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Try several initial guesses and local optimizations
    best_x = np.ones(N)
    best_val = _compute_ratio(best_x)
    
    # Generate diverse initial points
    rng = np.random.default_rng(42)
    
    initial_points = []
    # Uniform
    initial_points.append(np.ones(N))
    # Spike
    for i in range(N):
        spike = np.zeros(N)
        spike[i] = 1.0
        initial_points.append(spike)
    # Random
    for _ in range(20):
        initial_points.append(rng.random(N) + 1e-6)
    # Bimodal
    for _ in range(10):
        x = np.zeros(N)
        center = rng.integers(5, N-5)
        width = rng.integers(2, 10)
        x[center-width:center+width+1] = 1.0
        initial_points.append(x)
    # Triangular
    for _ in range(10):
        x = np.abs(np.arange(N) - N//2)
        x = N - x
        x = x / x.max()
        initial_points.append(x)
    # Gaussian-like
    for _ in range(10):
        t = np.linspace(-3, 3, N)
        x = np.exp(-t**2 / 2)
        initial_points.append(x)
    
    for x0 in initial_points:
        if time.time() > deadline - 5.0:
            break
        x0 = np.asarray(x0, dtype=np.float64)
        if np.any(x0 < 0):
            x0 = np.abs(x0)
        
        # Local optimization using BFGS
        try:
            res = minimize(
                _negative_ratio,
                x0,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8},
                timeout=30.0
            )
            x_opt = res.x
            if np.any(x_opt < 0):
                x_opt = np.abs(x_opt)
            val = _compute_ratio(x_opt)
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for robustness
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                _negative_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            x_opt = res.x
            if np.any(x_opt < 0):
                x_opt = np.abs(x_opt)
            val = _compute_ratio(x_opt)
            if val > best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Additional refinement: try coordinate ascent
    if time.time() < deadline - 10.0:
        for _ in range(5000):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = _compute_ratio(best_x)
                # Try perturbing coordinate i
                for delta in [0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 0.01, 0.05]:
                    trial = best_x.copy()
                    trial[i] = max(0.0, best_x[i] + delta)
                    new_val = _compute_ratio(trial)
                    if new_val > old_val + 1e-12:
                        best_x = trial
                        old_val = new_val
                        improved = True
                        break
            if not improved:
                break
    
    # Normalize to ensure nonnegative and return as list
    best_x = np.maximum(best_x, 0.0)
    if np.sum(best_x) < 1e-15:
        best_x = np.ones(N)
    
    return list(best_x)
# EVOLVE-BLOCK-END
