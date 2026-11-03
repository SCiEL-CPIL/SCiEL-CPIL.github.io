import numpy as np
import scipy.optimize as opt
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Generate test grid
    # Use a Fibonacci sphere for the grid too, with more points
    M = 2000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    G = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    def objective(flat_P):
        P = flat_P.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
        # Compute min angular distance for each grid point
        # Dot product: G (M,3) @ P.T (3,N) -> (M,N)
        dots = G @ P.T
        # Clip to [-1, 1] for acos
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distance
        dists = np.arccos(dots)
        # Min distance for each grid point
        min_dists = np.min(dists, axis=1)
        # Objective: sum of 4th powers to emphasize large distances
        return np.sum(min_dists ** 4)
    
    # Optimize
    best_P = P_init
    best_val = objective(P_init.flatten())
    
    # Try Powell optimization
    try:
        res = opt.minimize(objective, P_init.flatten(), method='Powell', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-8})
        if time.time() - start_time < deadline:
            if res.fun < best_val:
                best_P = res.x.reshape(N, 3)
                best_val = res.fun
    except Exception:
        pass
    
    # Try Nelder-Mead if time permits
    if time.time() - start_time < deadline - 20:
        try:
            res2 = opt.minimize(objective, best_P.flatten(), method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-8})
            if res2.fun < best_val:
                best_P = res2.x.reshape(N, 3)
                best_val = res2.fun
        except Exception:
            pass

    # Final normalization
    norms = np.linalg.norm(best_P, axis=1, keepdims=True)
    best_P = best_P / norms
    
    return best_P
