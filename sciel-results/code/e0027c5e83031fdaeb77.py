"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _energy_and_grad(x_flat):
    x = x_flat.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    # Compute pairwise differences and squared distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero and log(0) on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy = -sum_{i<j} log(dist_ij)
    # dist_ij = sqrt(dist_sq_ij)
    # log(dist_ij) = 0.5 * log(dist_sq_ij)
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient
    # grad_k = -sum_{j!=k} (x_k - x_j) / dist_sq_kj
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Vectorized gradient calculation
    # diff is (N, N, 3)
    # We want to sum over j (axis 1) of diff[k, j, :] * inv_dist_sq[k, j]
    # This is equivalent to einsum('kji,ki->kj', diff, inv_dist_sq) then sum over j?
    # No, grad_k is a vector of size 3.
    # grad[k] = - sum_j diff[k, j] * inv_dist_sq[k, j]
    # This is a matrix multiplication if we view diff as (N, N, 3) and inv_dist_sq as (N, N)
    # grad = -np.einsum('kji,ki->kj', diff, inv_dist_sq)
    # Wait, diff[k, j, m] * inv_dist_sq[k, j] summed over j.
    # Result is (N, 3).
    grad = -np.einsum('kji,ki->kj', diff, inv_dist_sq)
    
    # Project gradient onto tangent space of sphere
    # grad_tangent = grad - (grad . x_k) * x_k
    proj = np.sum(grad * x, axis=1, keepdims=True)
    grad = grad - proj * x
    
    return energy, grad.flatten()


def construct():
    best_energy = np.inf
    best_x = None
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Random inits
    rng = np.random.default_rng(42)
    for _ in range(9):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
        
    # Optimize from each start
    for x0 in inits:
        try:
            res = minimize(
                lambda x: _energy_and_grad(x)[0],
                x0.flatten(),
                jac=lambda x: _energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            e = res.fun
            if e < best_energy:
                best_energy = e
                best_x = res.x.reshape(N, 3)
        except Exception:
            continue
            
    # Final normalization
    if best_x is not None:
        norms = np.linalg.norm(best_x, axis=1, keepdims=True)
        best_x = best_x / norms
        return best_x
    else:
        # Fallback to Fibonacci
        return x_fib
# EVOLVE-BLOCK-END
