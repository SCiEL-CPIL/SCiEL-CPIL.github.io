"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Precompute pair indices for efficient energy calculation
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _energy_from_spherical(params):
        """
        params: array of shape (2*N,)
        First N are theta (polar angle, 0 to pi)
        Next N are phi (azimuthal angle, 0 to 2*pi)
        """
        theta = params[:N]
        phi = params[N:]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        X = np.column_stack([x, y, z])
        
        # Calculate distances
        diff = X[idx_i] - X[idx_j]
        dist = np.linalg.norm(diff, axis=1)
        
        # Avoid log(0)
        dist = np.maximum(dist, 1e-12)
        
        energy = -np.sum(np.log(dist))
        return energy

    def _local_minimize_spherical(params0, max_iter=500):
        """
        Run L-BFGS-B on spherical coordinates.
        """
        # Bounds for theta: [0, pi], for phi: [0, 2*pi]
        bounds = [(0, np.pi)] * N + [(0, 2 * np.pi)] * N
        
        # L-BFGS-B is efficient for this size
        res = minimize(
            _energy_from_spherical, 
            params0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
        )
        
        return res.x, res.fun

    def _fibonacci_init():
        """Generate initial spherical coordinates for Fibonacci sphere."""
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        # Convert back to spherical for optimizer
        theta = np.arccos(z)
        phi_sph = np.arctan2(y, x)
        # Ensure phi is in [0, 2*pi)
        phi_sph = np.mod(phi_sph, 2 * np.pi)
        
        params = np.concatenate([theta, phi_sph])
        return params

    def _random_init(seed):
        """Generate random initial spherical coordinates."""
        rng = np.random.default_rng(seed)
        # Uniform on sphere: z uniform in [-1,1], phi uniform in [0,2pi]
        z = rng.uniform(-1, 1, N)
        phi = rng.uniform(0, 2 * np.pi, N)
        theta = np.arccos(z)
        
        params = np.concatenate([theta, phi])
        return params

    # Generate diverse initializations
    initializations = []
    initializations.append(_fibonacci_init())
    
    # Add several random restarts
    for s in range(40):
        initializations.append(_random_init(42 + s))
        
    best_params = initializations[0].copy()
    best_energy = float('inf')
    
    # Run local minimization from each initialization
    for i, init_params in enumerate(initializations):
        if time.time() > deadline - 2.0:
            break
            
        # We can reduce max_iter for later restarts if time is tight, 
        # but L-BFGS-B is fast. Let's give them all a fair chance.
        # Maybe use a smaller max_iter for random ones if we have many?
        # 500 iterations is usually enough for 200 variables.
        
        try:
            res_params, res_energy = _local_minimize_spherical(init_params, max_iter=300)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_params = res_params
        except Exception:
            continue

    # Convert best parameters back to Cartesian coordinates
    theta = best_params[:N]
    phi = best_params[N:]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    X = np.column_stack([x, y, z])
    
    # Normalize to ensure unit sphere (fix numerical drift)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X = X / norms
    
    # Final check: ensure no points are too close
    # If points are closer than 1e-9, we might need to perturb them, 
    # but the optimizer should avoid this.
    
    return X
# EVOLVE-BLOCK-END
