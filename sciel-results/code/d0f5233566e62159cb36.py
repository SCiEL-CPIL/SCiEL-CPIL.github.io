import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def _log_energy(x):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = x.shape[0]
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    energy = -0.5 * math.fsum(np.log(d_sq_upper).tolist())
    return energy

def _energy_and_grad(x_flat, start_time, time_limit):
    """Compute energy and projected gradient for the sphere-constrained problem."""
    if time.time() - start_time > time_limit - 5.0:
        raise TimeoutError()
    
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_upper = dist_sq[mask]
    
    energy = -0.5 * np.sum(np.log(dist_sq_upper))
    
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / dist_sq[mask]
    
    sum_W = W.sum(axis=1)
    grad_raw = -(x * sum_W[:, None] - (W @ x.T).T)
    
    dot_products = np.sum(grad_raw * x, axis=1)
    grad_proj = grad_raw - dot_products[:, None] * x
    
    return energy, grad_proj.flatten()

def _polish(x, start_time, time_limit, maxiter=5000, gtol=1e-14, ftol=1e-16, maxls=100):
    """High-precision polish pass with very tight tolerances."""
    try:
        result = minimize(
            _energy_and_grad,
            x.flatten(),
            method='L-BFGS-B',
            jac=True,
            args=(start_time, time_limit),
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': maxls}
        )
        x_opt = result.x.reshape(N, 3)
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_opt = x_opt / norms
        return x_opt
    except Exception:
        return x

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Start with Fibonacci sphere
    x_init = _fibonacci_sphere(N)
    best_x = x_init.copy()
    best_energy = _log_energy(best_x)
    
    # Helper to run optimization and update best
    def run_optimization(x_start, maxiter, gtol=1e-10, ftol=1e-12):
        nonlocal best_x, best_energy
        try:
            result = minimize(
                _energy_and_grad,
                x_start.flatten(),
                method='L-BFGS-B',
                jac=True,
                args=(start_time, time_limit),
                options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
            )
            
            if result.fun < best_energy:
                x_opt = result.x.reshape(N, 3)
                norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                x_opt = x_opt / norms
                
                e_opt = _log_energy(x_opt)
                if e_opt < best_energy:
                    best_x = x_opt
                    best_energy = e_opt
        except (TimeoutError, Exception):
            pass

    # 1. Initial optimization from Fibonacci start (moderate iterations for screening)
    run_optimization(x_init, 1500, gtol=1e-10, ftol=1e-12)
    
    # 2. Multiple restarts with varying noise scales to escape local minima
    rng = np.random.default_rng(42)
    
    # Improved noise schedule: mix of small perturbations (refinement) and larger ones (escape)
    noise_scales = [0.08, 0.12, 0.05, 0.15, 0.03, 0.20, 0.06, 0.10, 0.04, 0.18,
                    0.07, 0.13, 0.05, 0.25, 0.03, 0.09, 0.11, 0.04, 0.16, 0.06]
    
    # Adaptive restart loop: continue until time budget exhausted
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        # Reserve time for final polish (at least 15s)
        if elapsed > time_limit - 15.0:
            break
        # Estimate time per restart (~3s average), ensure we can fit at least one more
        if elapsed + 3.0 > time_limit - 15.0:
            break
            
        scale = noise_scales[restart_count % len(noise_scales)]
        noise = rng.normal(0, scale, size=(N, 3))
        x_perturbed = best_x + noise
        norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_perturbed = x_perturbed / norms
        
        # Use fewer iterations for restarts (screening)
        run_optimization(x_perturbed, 800, gtol=1e-9, ftol=1e-11)
        restart_count += 1
    
    # 3. First high-precision polish pass on the best candidate
    if time.time() - start_time < time_limit - 25.0:
        polished = _polish(best_x, start_time, time_limit, maxiter=3000, gtol=1e-13, ftol=1e-15, maxls=80)
        e_polished = _log_energy(polished)
        if e_polished < best_energy:
            best_x = polished
            best_energy = e_polished
    
    # 4. Second ultra-tight polish pass if time permits
    if time.time() - start_time < time_limit - 10.0:
        polished2 = _polish(best_x, start_time, time_limit, maxiter=5000, gtol=1e-15, ftol=1e-16, maxls=100)
        e_polished2 = _log_energy(polished2)
        if e_polished2 < best_energy:
            best_x = polished2
            best_energy = e_polished2
    
    # Final validation
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_x = best_x / norms
    
    # Check for very close points to ensure validity
    dist_sq = np.sum((best_x[:, None, :] - best_x[None, :, :]) ** 2, axis=2)
    min_dist_sq = np.min(dist_sq[np.triu_indices(N, k=1)])
    if min_dist_sq < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_x

# EVOLVE-BLOCK-END
