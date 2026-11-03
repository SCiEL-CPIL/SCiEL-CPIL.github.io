"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random jitter to break symmetry and avoid bad local minima
    # np.random.seed(42) for reproducibility in evaluation, but random is fine
    x0 = x0 + np.random.normal(0, 0.01, x0.shape)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

    # 2. Define Energy and Gradient with Penalty
    # Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    # Penalty: lambda * sum (||x_i||^2 - 1)^2
    
    def energy_and_grad(x_flat, lam):
        x = x_flat.reshape(N, 3)
        
        # Compute pairwise distances
        # diff[i, j, k] = x[i, k] - x[j, k]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero (should not happen with jitter, but safe)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Log Energy: -log(dist)
        # dist = sqrt(dist_sq)
        # log(1/dist) = -log(dist) = -0.5 * log(dist_sq)
        log_dist = 0.5 * np.log(dist_sq)
        
        # Mask diagonal (i == j)
        mask = ~np.eye(N, dtype=bool)
        
        # Sum over i < j
        # The matrix is symmetric. We sum upper triangle.
        triu_indices = np.triu_indices(N, 1)
        energy = -np.sum(log_dist[triu_indices])
        
        # Penalty Term
        norms_sq = np.sum(x * x, axis=1)
        penalty_val = lam * np.sum((norms_sq - 1.0) ** 2)
        
        total_energy = energy + penalty_val
        
        # Gradient Calculation
        # d(energy)/dx
        # E = sum_{i<j} -log(||x_i - x_j||)
        # dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # Note: The derivative of -log(r) is -1/r * dr/dx = -(x_k-x_j)/r * 1/r = -(x_k-x_j)/r^2
        # Wait, log(1/r) = -log(r). d(-log r)/dr = -1/r. dr/dx = (x_i-x_j)/r.
        # So grad is -(x_i-x_j)/r^2.
        # Let's verify signs. If points get closer, r decreases, log(1/r) increases. 
        # We want to minimize energy. So we want to move points apart.
        # Force should be repulsive.
        # Gradient of Energy w.r.t x_i points in direction of increasing energy.
        # So we move in -grad direction.
        # Grad_i = sum_{j != i} (x_i - x_j) / dist_sq_ij ?
        # Let's check: E = -log(r). dE/dx_i = -1/r * (x_i-x_j)/r = -(x_i-x_j)/r^2.
        # This vector points from j to i? No, x_i - x_j points from j to i.
        # If x_i is far, vector is large. 
        # Let's just compute the matrix of gradients.
        
        # grad_energy[i, k] = sum_{j != i} (x_i - x_j)_k / dist_sq_ij
        # We can compute this using matrix operations.
        # inv_dist_sq = 1 / dist_sq
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0) # Exclude self
        
        # grad[i, k] = sum_j diff[i, j, k] * inv_dist_sq[i, j]
        # diff[i, j, k] = x_i - x_j
        grad_energy = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        # Penalty Gradient
        # P = lam * sum (||x_i||^2 - 1)^2
        # dP/dx_i = lam * 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * lam * (||x_i||^2 - 1) * x_i
        grad_penalty = 4.0 * lam * (norms_sq[:, np.newaxis] - 1.0) * x
        
        total_grad = grad_energy + grad_penalty
        
        return total_energy, total_grad.flatten()

    # 3. Optimization Stages
    # Stage 1: Moderate penalty to find good configuration
    res1 = minimize(
        lambda x: energy_and_grad(x, 100.0)[0],
        x0.flatten(),
        jac=lambda x: energy_and_grad(x, 100.0)[1],
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    x1 = res1.x.reshape(N, 3)
    
    # Stage 2: High penalty to enforce sphere constraint strictly
    res2 = minimize(
        lambda x: energy_and_grad(x, 10000.0)[0],
        x1.flatten(),
        jac=lambda x: energy_and_grad(x, 10000.0)[1],
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14}
    )
    
    x2 = res2.x.reshape(N, 3)
    
    # Final Normalization to ensure unit sphere
    x_final = x2 / np.linalg.norm(x2, axis=1, keepdims=True)
    
    return x_final
# EVOLVE-BLOCK-END
