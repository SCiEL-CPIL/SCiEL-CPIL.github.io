"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy(pts):
        """Compute logarithmic energy using math.fsum for accuracy."""
        e = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-15:
                    return float('inf')
                e.append(math.log(1.0 / d))
        return math.fsum(e)
    
    # L-BFGS-B optimization using numpy/scipy
    from scipy.optimize import minimize
    
    def flatten_points(pts):
        return pts.flatten()
    
    def unflatten_points(flat):
        pts = flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        return pts
    
    def objective(flat):
        pts = unflatten_points(flat)
        return energy(pts)
    
    def grad(flat):
        pts = unflatten_points(flat)
        # Numerical gradient (could be improved with analytical gradient)
        eps = 1e-7
        g = np.zeros_like(flat)
        for k in range(len(flat)):
            orig = flat[k]
            flat[k] = orig + eps
            f_plus = objective(flat)
            flat[k] = orig
            g[k] = (f_plus - objective(flat)) / eps
            flat[k] = orig
        return g
    
    best_energy = energy(points)
    best_points = points.copy()
    
    # Try a few different initializations
    inits = [points.copy()]
    
    # Add some random perturbations
    for trial in range(5):
        if time.time() - start_time > time_limit - 30:
            break
        rng = np.random.RandomState(trial * 123 + 42)
        perturbed = points + rng.randn(N, 3) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        inits.append(perturbed)
    
    for init in inits:
        if time.time() - start_time > time_limit - 10:
            break
        
        flat_init = init.flatten()
        
        try:
            res = minimize(
                objective,
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
                timeout=20
            )
            
            if res.success or res.fun < best_energy:
                opt_points = unflatten_points(res.x)
                e = energy(opt_points)
                if e < best_energy:
                    best_energy = e
                    best_points = opt_points.copy()
        except Exception:
            continue
    
    # Final check
    if time.time() - start_time > time_limit:
        return best_points
    
    # One more polish with tighter tolerance
    try:
        flat_best = best_points.flatten()
        res2 = minimize(
            objective,
            flat_best,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30},
            timeout=5
        )
        if res2.fun < best_energy:
            opt_points2 = unflatten_points(res2.x)
            e2 = energy(opt_points2)
            if e2 < best_energy:
                best_points = opt_points2
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
