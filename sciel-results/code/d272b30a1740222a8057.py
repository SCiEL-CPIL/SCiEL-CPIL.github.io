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
    
    # Precompute pair indices for efficient energy/gradient calculation
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)
    
    def _energy_and_grad(X):
        """Compute log energy and Euclidean gradient for points on sphere."""
        # X: (N, 3) unit vectors
        diff = X[idx_i] - X[idx_j]  # (n_pairs, 3)
        dist_sq = np.sum(diff * diff, axis=1)  # (n_pairs,)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero
        dist = np.maximum(dist, 1e-12)
        
        # Energy: sum log(1/d) = -sum log(d)
        energy = -np.sum(np.log(dist))
        
        # Gradient: d/dX_i [sum_{j != i} log(1/|X_i - X_j|)]
        # = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
        grad = np.zeros((N, 3))
        inv_d2 = 1.0 / dist_sq
        # grad_i += (X_i - X_j) / d^2
        np.add.at(grad, idx_i, diff * inv_d2[:, np.newaxis])
        # grad_j += (X_j - X_i) / d^2 = -diff / d^2
        np.add.at(grad, idx_j, -diff * inv_d2[:, np.newaxis])
        
        return energy, grad

    def _project_to_sphere(X):
        """Project points to the unit sphere."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return X / norms

    def _local_minimize(X0, max_iter=1000, tol=1e-10):
        """L-BFGS-B local minimization with sphere projection."""
        X = X0.copy()
        
        def objective_and_grad(X_flat):
            X = X_flat.reshape(N, 3)
            # Project to sphere to maintain constraint
            X = _project_to_sphere(X)
            e, g = _energy_and_grad(X)
            # Flatten gradient
            g_flat = g.reshape(-1)
            return e, g_flat
        
        # Use L-BFGS-B
        result = minimize(
            objective_and_grad,
            X.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
        )
        
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        return X_opt, result.fun

    def _fibonacci_init():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def _random_init(seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, 3))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    # Start with Fibonacci initialization
    best_X = _fibonacci_init()
    best_energy = _energy_and_grad(best_X)[0]
    
    # Try multiple random initializations with basin hopping
    rng = np.random.default_rng(42)
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial point
        X_init = _random_init(restart * 1000 + 42)
        
        # Local minimization
        X_opt, e_opt = _local_minimize(X_init)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Final refinement with more iterations on the best solution
    X_final, e_final = _local_minimize(best_X, max_iter=2000, tol=1e-12)
    if e_final < best_energy:
        best_energy = e_final
        best_X = X_final
    
    # Ensure all points are distinct and on the unit sphere
    best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
