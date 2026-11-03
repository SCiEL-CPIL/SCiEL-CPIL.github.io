"""Minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
# The evaluator kills at 150s. We should aim to finish by 145s to be safe.
DEADLINE = 145.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _log_energy_fsum(x):
    """
    Compute logarithmic energy using math.fsum for high precision,
    matching the evaluator's metric.
    E = sum_{i<j} log(1/|x_i - x_j|)
    """
    n = x.shape[0]
    # Compute squared distances
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    # Avoid numerical issues
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    
    # log(1/d) = -log(d)
    # We need to sum these values with math.fsum
    # Converting to list for math.fsum
    log_vals = -np.log(d)
    return math.fsum(log_vals.tolist())

def _log_energy_np(x):
    """
    Fast energy calculation for optimization loops (using numpy sum).
    Used for gradient descent steps where relative precision is less critical
    than speed, but we will use fsum for final comparison.
    """
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return -np.sum(np.log(d))

def _log_energy_grad(x):
    """
    Compute gradient of logarithmic energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    coeff = 1.0 / dist_sq
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_{j != i} coeff[i,j] * (x_j - x_i)
    # = sum_j coeff[i,j] x_j - x_i sum_j coeff[i,j]
    coeff_sum = coeff.sum(axis=1, keepdims=True)
    grad = coeff @ x - coeff_sum * x
    
    return grad

def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms

def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x

def _golden_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _sobol_sphere_points(n, seed=0):
    """Generate Sobol sequence points mapped to the unit sphere."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
        # Generate n points
        u = sampler.random(n)
        # Map to sphere:
        # z is uniform in [-1, 1]
        # phi is uniform in [0, 2*pi]
        z = 2.0 * u[:, 2] - 1.0
        phi = 2.0 * math.pi * u[:, 0]
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.column_stack([x, y, z])
    except Exception:
        # Fallback to random if Sobol fails
        rng = np.random.default_rng(seed)
        return _random_sphere_points(n, rng)

def _optimize_lbfgs(x0, time_budget, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        # Use fast energy for optimization
        e = _log_energy_np(x)
        g = _log_energy_grad(x)
        g_tangent = _tangent_project(g, x)
        return e, g_tangent.flatten()
    
    x_flat = x0.flatten()
    
    best_x = x0.copy()
    # Use fast energy for tracking best during optimization
    best_e = _log_energy_np(x0)
    
    try:
        res = minimize(
            objective_and_grad,
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        # Check if this is better using fast metric
        e_opt_fast = _log_energy_np(x_opt)
        if e_opt_fast < best_e:
            best_e = e_opt_fast
            best_x = x_opt.copy()
    except Exception:
        pass
    
    # If time remains, do a second pass with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.6:
        try:
            res = minimize(
                objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol * 0.01, 'gtol': gtol * 0.01}
            )
            x_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt_fast = _log_energy_np(x_opt)
            if e_opt_fast < best_e:
                best_e = e_opt_fast
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Golden spirals with different offsets
    for offset in [0.0, 1.0, 2.0, 0.5, 3.0]:
        inits.append(_golden_spiral(N, seed_offset=offset))
        
    # 2. Sobol sequences (better uniform coverage)
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        inits.append(_sobol_sphere_points(N, seed=seed))
        
    # 3. Random points
    for seed in [10, 20, 30, 40]:
        r = np.random.default_rng(seed)
        inits.append(_random_sphere_points(N, r))
        
    # 4. Perturbed Golden Spiral
    for scale in [0.1, 0.05]:
        perturbed = _golden_spiral(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))

    best_x = None
    best_e_fsum = float('inf')
    
    num_inits = len(inits)
    
    for i, x0 in enumerate(inits):
        if _remaining_time() < 15.0:
            break
        
        # Allocate time budget
        # We want to spend roughly equal time on each, but prioritize early ones?
        # Or just divide remaining time?
        # Let's give each init a fixed reasonable time, say 10s, but cap by remaining.
        time_budget = min(10.0, _remaining_time() * 0.8)
        
        x_opt = _optimize_lbfgs(x0, time_budget, maxiter=3000, ftol=1e-15, gtol=1e-12)
        
        # Evaluate with high precision (fsum) to compare
        e_opt_fsum = _log_energy_fsum(x_opt)
        
        if e_opt_fsum < best_e_fsum:
            best_e_fsum = e_opt_fsum
            best_x = x_opt.copy()
            
    if best_x is None:
        best_x = _golden_spiral(N)
        best_e_fsum = _log_energy_fsum(best_x)
        
    # Final Polish Phase
    # Use remaining time for a very tight optimization on the best candidate found
    if _remaining_time() > 10.0:
        # We can run a long L-BFGS with very tight tolerances
        # Note: L-BFGS-B might not converge to machine precision easily, but we can try
        x_polish = _optimize_lbfgs(best_x, _remaining_time() * 0.9, maxiter=10000, ftol=1e-16, gtol=1e-14)
        
        e_polish_fsum = _log_energy_fsum(x_polish)
        if e_polish_fsum < best_e_fsum:
            best_x = x_polish
            best_e_fsum = e_polish_fsum
            
    # Final validation and projection
    best_x = _project_to_sphere(best_x)
    
    # Check for close points and fix if necessary
    # If any pair is < 1e-9, the score is invalid.
    # We should verify this.
    dists = np.sqrt(np.sum((best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 1e-9:
        # Nudge points apart
        # This is a rare event in optimized configurations, but if it happens,
        # we need to handle it.
        for i in range(N):
            for j in range(i+1, N):
                if dists[i,j] < 1e-9:
                    # Move j slightly
                    noise = rng.standard_normal(3) * 1e-8
                    best_x[j] = best_x[j] + noise
                    best_x[j] = _project_to_sphere(best_x[j])
        # Re-project all to be safe
        best_x = _project_to_sphere(best_x)
        
    # Ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = _golden_spiral(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
