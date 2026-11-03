import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the unit sphere.
    points: (N, 3) array of points (not necessarily normalized).
    Returns: energy (float), gradient (N, 3) array (tangential).
    """
    x = _normalize(points)
    
    # Compute pairwise distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # dist = sqrt(dist_sq)
    # log(dist) = 0.5 * log(dist_sq)
    # So energy = -0.5 * sum_{i<j} log(dist_sq)
    
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    d2 = dist_sq[iu]
    
    # Clip to avoid log(0) if points are too close (should be rare with good init)
    d2 = np.maximum(d2, 1e-30)
    
    energy = -0.5 * np.sum(np.log(d2))
    
    # Gradient calculation
    # grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # This is the Euclidean gradient.
    # We need the tangential component: grad_t = grad_e - (grad_e . x_k) x_k
    
    grad = np.zeros_like(x)
    
    # Efficient calculation of gradient
    # For each pair (i, j), the contribution to grad[i] is (x_i - x_j) / d2_ij
    # and to grad[j] is (x_j - x_i) / d2_ij
    
    # Using the upper triangle indices
    i_idx, j_idx = iu
    d2_ij = d2[i_idx, j_idx]
    
    # Vector differences
    v = x[i_idx] - x[j_idx]
    
    # Coefficients
    coeff = 1.0 / d2_ij
    
    # Accumulate gradients
    # grad[i] += v * coeff
    # grad[j] -= v * coeff
    np.add.at(grad, i_idx, v * coeff[:, np.newaxis])
    np.add.at(grad, j_idx, -v * coeff[:, np.newaxis])
    
    # Project to tangent space
    # grad_t = grad - (grad . x) x
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad_t = grad - dot * x
    
    return energy, grad_t

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization function wrapper for scipy
    def fun(x_flat):
        x = x_flat.reshape(N, 3)
        e, _ = _energy_and_grad(x)
        return e
    
    def jac(x_flat):
        x = x_flat.reshape(N, 3)
        _, g = _energy_and_grad(x)
        return g.flatten()
    
    best_x = x0.copy()
    best_e = fun(best_x.flatten())
    
    # Run L-BFGS-B
    try:
        res = minimize(
            fun, 
            best_x.flatten(), 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if res.success or res.fun < best_e:
            best_x = res.x.reshape(N, 3)
            best_e = res.fun
            
        # Check time
        if time.time() > deadline:
            return _normalize(best_x)
            
        # If time remains, try a few random perturbations to escape local minima
        # The Fibonacci spiral is a strong start, but slight noise might help find a better basin.
        # However, random restarts from scratch are risky. Perturbing the current best is safer.
        
        rng = np.random.default_rng(42)
        
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Add small noise
            noise = rng.normal(0, 0.01, size=(N, 3))
            x_pert = best_x + noise
            
            # Normalize
            x_pert = _normalize(x_pert)
            
            # Optimize from perturbed start
            try:
                res2 = minimize(
                    fun, 
                    x_pert.flatten(), 
                    jac=jac, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res2.fun < best_e:
                    best_e = res2.fun
                    best_x = res2.x.reshape(N, 3)
            except Exception:
                pass
                
    except Exception:
        pass
        
    # Final normalization
    return _normalize(best_x)
# EVOLVE-BLOCK-END
