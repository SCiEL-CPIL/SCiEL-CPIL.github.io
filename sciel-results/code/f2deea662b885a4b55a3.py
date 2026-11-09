import numpy as np
from math import fsum

N = 150

# EVOLVE-BLOCK-START

def compute_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)"""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    # Only compute for upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def construct():
    # 1. Initialize with Fibonacci spiral (good starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Gradient Descent with adaptive step size and line search
    num_steps = 10000
    initial_lr = 0.1
    min_lr = 1e-6
    
    best_energy = compute_energy(x)
    
    for step in range(num_steps):
        # Calculate forces
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Set diagonal to infinity to avoid self-interaction division by zero
        np.fill_diagonal(dist_sq, np.inf)
        
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
        
        force = diff * inv_dist_sq[:, :, np.newaxis]
        total_force = np.sum(force, axis=1)
        
        # Project force onto tangent plane
        dot_products = np.sum(total_force * x, axis=1)
        tangent_force = total_force - dot_products[:, np.newaxis] * x
        
        # Adaptive learning rate with exponential decay
        lr = max(initial_lr * (0.99 ** (step / 10.0)), min_lr)
        
        # Try multiple step sizes (backtracking line search)
        best_x = x
        best_e = best_energy
        
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05]:
            candidate = x + lr * alpha * tangent_force
            norms = np.linalg.norm(candidate, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            candidate = candidate / norms
            
            # Check validity
            if not np.all(np.isfinite(candidate)):
                continue
            
            e = compute_energy(candidate)
            if e < best_e - 1e-12:
                best_x = candidate
                best_e = e
        
        if best_e < best_energy - 1e-12:
            x = best_x
            best_energy = best_e
        
        # Convergence check
        if step > 500 and step % 50 == 0:
            # Check if gradient norm is small
            grad_norm = np.linalg.norm(tangent_force)
            if grad_norm < 1e-8:
                break
    
    # Final normalization and validation
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    if not np.all(np.isfinite(x)):
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return x

# EVOLVE-BLOCK-END
