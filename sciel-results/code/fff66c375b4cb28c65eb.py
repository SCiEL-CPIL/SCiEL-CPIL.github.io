import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_smooth_max_covering_radius(X, grid, p=20):
    """
    X: (N, 3) array of points
    grid: (K, 3) array of test points
    p: power for smooth max approximation
    """
    # Compute squared distances between all grid points and all X points
    # dist^2 = |y|^2 + |x|^2 - 2 y.x = 1 + 1 - 2 y.x = 2 - 2 y.x
    # Since we want angular distance, we can work with cos(theta) or theta directly.
    # Minimizing max theta is equivalent to minimizing max (1 - cos(theta)) or similar.
    # Let's use theta = arccos(y.x).
    
    # Vectorized dot product: (K, 1, 3) dot (1, N, 3) -> (K, N)
    dots = grid @ X.T  # (K, N)
    
    # Clip to [-1, 1] for numerical stability in arccos
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances
    dists = np.arccos(dots)  # (K, N)
    
    # Min distance for each grid point to any X point
    min_dists = np.min(dists, axis=1)  # (K,)
    
    # Smooth max approximation: (1/K * sum(d^p))^(1/p)
    # Using p=20 or higher makes it close to max
    if p == 0:
        return np.max(min_dists)
    else:
        return (np.mean(min_dists ** p)) ** (1.0 / p)

def _optimize_covering(X_init, grid, p=20, time_limit=100):
    start_time = time.time()
    
    # Flatten X for optimizer
    x0 = X_init.flatten()
    
    def objective(flat_x):
        X = flat_x.reshape(N, 3)
        # Normalize X to ensure it stays on sphere during optimization steps
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X = X / norms
        return _compute_smooth_max_covering_radius(X, grid, p)
    
    # Use L-BFGS-B
    # We don't have bounds in Cartesian coordinates easily, but normalization handles it.
    # However, L-BFGS-B assumes bounded variables. 
    # Alternative: Optimize on the sphere using tangent space? 
    # Simpler: Just use Nelder-Mead? It's derivative-free but slow for 150 vars.
    # Let's stick to L-BFGS-B but be aware it might step off the sphere. 
    # The normalization inside the objective function makes the landscape non-smooth at the origin, 
    # but since we start on the sphere and step sizes are small, it usually stays close.
    
    # A better approach for sphere constraints is to use scipy.optimize.minimize with method='SLSQP' 
    # and equality constraints x^2+y^2+z^2=1 for each point. 
    # But 50 constraints might be slow.
    
    # Let's try a different representation: optimize the 3D coordinates and rely on the 
    # normalization in the objective. This is a common hack.
    
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return X_init # Return best effort or initial if timeout? 
        # Actually, minimize returns best found so far.
    
    best_X = res.x.reshape(N, 3)
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # 1. Initial Guess: Fibonacci Spiral
    X = _generate_fibonacci(N)
    
    # 2. Generate Test Grid for Objective Function
    # 20,000 points is a good balance between accuracy and speed
    K = 20000
    grid = _generate_fibonacci(K)
    
    # 3. Optimization Phase 1: Smooth Max with p=10 (broader)
    if time.time() < deadline - 30:
        X = _optimize_covering(X, grid, p=10, time_limit=30)
    
    # 4. Optimization Phase 2: Smooth Max with p=50 (sharper, closer to true max)
    if time.time() < deadline - 15:
        X = _optimize_covering(X, grid, p=50, time_limit=15)
        
    # 5. Final Refinement: Nelder-Mead for local fine-tuning? 
    # Nelder-Mead can be effective for the final few steps on the non-smooth max function.
    if time.time() < deadline - 5:
        # Use the true max distance as objective for Nelder-Mead
        def true_objective(flat_x):
            X_temp = flat_x.reshape(N, 3)
            X_temp = X_temp / np.linalg.norm(X_temp, axis=1, keepdims=True)
            # Use a smaller grid for speed in Nelder-Mead iterations? 
            # Or just use the same grid.
            return _compute_smooth_max_covering_radius(X_temp, grid, p=100)
            
        res = minimize(true_objective, X.flatten(), method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < true_objective(X.flatten()):
            X = res.x.reshape(N, 3)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            
    # Final normalization
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    return X

# EVOLVE-BLOCK-END
