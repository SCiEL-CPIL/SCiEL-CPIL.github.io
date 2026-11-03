import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Helper to generate Fibonacci spiral
    def fibonacci_spiral(offset=0.5):
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z)) # max to avoid negative due to floating point
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pts = np.column_stack((x, y, z))
        return pts

    # Objective function
    def energy(params):
        pts = params.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # We only need i < j
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        
        # Avoid log(0) and ensure stability
        # Using a small epsilon to prevent log(0)
        d = np.clip(d, 1e-8, None)
        
        # Energy = sum log(1/d) = -sum log(d)
        return -np.sum(np.log(d))
        
    bounds = [(-10, 10)] * (3 * N)
    
    best_energy = np.inf
    best_pts = None
    
    # Multi-start strategy
    num_starts = 5
    
    for k in range(num_starts):
        if k == 0:
            # Start 0: Pure Fibonacci
            pts0 = fibonacci_spiral(offset=0.5)
        else:
            # Start k: Fibonacci with noise
            pts0 = fibonacci_spiral(offset=0.5)
            # Add small random noise
            noise = np.random.normal(0, 0.05, (N, 3))
            pts0 = pts0 + noise
            # Normalize
            norms = np.linalg.norm(pts0, axis=1, keepdims=True)
            pts0 = pts0 / norms
            
        x0 = pts0.flatten()
        
        res = minimize(energy, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pts = res.x
    
    # Final polish with more iterations on the best result
    if best_pts is not None:
        res_final = minimize(energy, best_pts, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if res_final.fun < best_energy:
            best_pts = res_final.x
            best_energy = res_final.fun

    # Convert best_params back to Cartesian and normalize
    pts = best_pts.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    return pts
