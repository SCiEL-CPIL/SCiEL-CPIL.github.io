import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    n = 14

    def params_to_points(params):
        theta = params[:n]
        phi = params[n:]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))

    def energy(params):
        pts = params_to_points(params)
        dists = pdist(pts, 'euclidean')
        # Small epsilon to avoid div by zero
        dists = np.maximum(dists, 1e-12)
        return np.sum(1.0 / dists**2)

    # Initial guess: Fibonacci Sphere
    i = np.arange(n)
    phi_init = np.arccos(1 - 2 * (i + 0.5) / n)
    theta_init = np.pi * (1 + 5**0.5) * i
    init_params = np.concatenate([theta_init, phi_init])
    
    best_params = init_params
    best_energy = energy(init_params)

    # Run optimization multiple times with different random restarts if time permits
    np.random.seed(123)
    
    while time.time() < deadline:
        # Perturb initial guess slightly for diversity in subsequent runs
        if np.array_equal(best_params, init_params):
            current_init = init_params.copy()
        else:
            current_init = best_params + np.random.normal(0, 0.1, size=2*n)
            # Wrap angles? Nelder-Mead works on the parameter space directly.
            # Spherical coords are periodic, which can cause issues for Nelder-Mead.
            # It's better to use Cartesian coordinates with a constraint, or accept that Nelder-Mead
            # might get stuck at the poles.
            # Let's stick to Nelder-Mead on spherical coords for simplicity, 
            # but restart from best_params + noise.
            
        res = minimize(energy, current_init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
        else:
            # If no improvement, break or continue with larger noise?
            # For now, break if no improvement for a while?
            # Simple: just loop until deadline.
            pass

    pts = params_to_points(best_params)
    
    # Optional: Final polish with exact metric optimization?
    # The energy proxy is good.
    
    return pts
