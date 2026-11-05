"""Combined program: spectral/FIR design + gradient refinement for autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _gradient(h):
    """Numerical gradient of the objective."""
    eps = 1e-7
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    conv = np.convolve(h, h, mode='full')
    f0 = abs(2.0 * N * np.max(conv) / (s * s))
    
    grad = np.zeros_like(h)
    for i in range(N):
        h_p = h.copy()
        h_p[i] += eps
        s_p = np.sum(h_p)
        if abs(s_p) < 1e-12:
            grad[i] = 0
            continue
        conv_p = np.convolve(h_p, h_p, mode='full')
        f_p = abs(2.0 * N * np.max(conv_p) / (s_p * s_p))
        grad[i] = (f_p - f0) / eps
    return grad

def _gaussian_bump(width, height, baseline):
    """Generate a Gaussian bump minus baseline."""
    x = np.arange(N)
    center = N / 2.0
    g = height * np.exp(-0.5 * ((x - center) / width) ** 2)
    return g - baseline

def _parametric_objective(params):
    """Objective for parametric family: width, height, baseline."""
    width, height, baseline = params
    if width < 1.0 or width > N/2:
        return 1e10
    if height < 0 or baseline < 0:
        return 1e10
    h = _gaussian_bump(width, height, baseline)
    return _evaluate(h)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Parametric search over Gaussian bump family
    # Try a grid of (width, height, baseline) combinations
    widths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    heights = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    baselines = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    for w in widths:
        if time.time() > deadline - 30.0:
            break
        for ht in heights:
            if time.time() > deadline - 30.0:
                break
            for bl in baselines:
                h = _gaussian_bump(w, ht, bl)
                val = _evaluate(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    # Phase 2: Local gradient descent refinement from best parametric result
    if time.time() < deadline - 20.0:
        h_curr = best_h.copy()
        # Normalize to keep sum reasonable
        s = np.sum(h_curr)
        if abs(s) > 1e-6:
            h_curr = h_curr / s * 10.0  # Scale so sum ~ 10
        
        for step in range(200):
            if time.time() > deadline - 15.0:
                break
            
            grad = _gradient(h_curr)
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                break
            
            # Adaptive step size
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12:
                break
            step_size = 0.01 / norm_grad
            
            h_new = h_curr - step_size * grad
            
            # Ensure finite
            if not np.all(np.isfinite(h_new)):
                break
            
            val_new = _evaluate(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            if val_new < _evaluate(h_curr):
                h_curr = h_new
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-10:
                    break
    
    # Phase 3: Try a few random perturbations around best
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            h_pert = best_h + 0.01 * rng.standard_normal(N)
            val_pert = _evaluate(h_pert)
            if val_pert < best_val:
                best_val = val_pert
                best_h = h_pert.copy()
    
    return best_h
# EVOLVE-BLOCK-END
