import numpy as np
import time

N = 150

def fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def energy_and_grad(x):
    # x: (N, 3)
    # Compute pairwise squared distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clamp distances to avoid log(0) and division by zero
    # 1e-12 is small enough to not affect energy significantly for valid configs
    # but large enough to prevent numerical instability
    dist_sq_clamped = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq_clamped)
    
    # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    iu = np.triu_indices(N, k=1)
    e = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (x_i - x_j) / dist_sq_ij
    # coeff_ij = 1 / dist_sq_ij
    coeff = np.zeros_like(dist_sq_clamped)
    coeff[dist_sq_clamped > 1e-12] = 1.0 / dist_sq_clamped[dist_sq_clamped > 1e-12]
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = x_i * sum_j coeff_ij - sum_j coeff_ij * x_j
    sum_coeff = np.sum(coeff, axis=1)
    grad = x * sum_coeff[:, np.newaxis] - np.sum(coeff[:, :, np.newaxis] * x[np.newaxis, :, :], axis=1)
    
    # Project gradient onto tangent space of the sphere
    # grad_proj = grad - (grad . x) x
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad_proj = grad - dot * x
    
    return e, grad_proj

def local_optimize(x, max_iters=2000, tol=1e-12):
    x = x.copy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    e, g = energy_and_grad(x)
    best_e = e
    best_x = x.copy()
    
    lr = 1e-3
    momentum = 0.9
    v = np.zeros_like(x)
    
    for i in range(max_iters):
        # Update velocity
        v = momentum * v - lr * g
        
        # Step
        x_new = x + v
        
        # Retraction (normalize to unit sphere)
        x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
        
        # Check energy
        e_new, g_new = energy_and_grad(x_new)
        
        if e_new < best_e:
            best_e = e_new
            best_x = x_new.copy()
            x = x_new
            g = g_new
            # If improving, we can keep lr or increase slightly
            # lr = min(lr * 1.05, 1e-2) 
        else:
            # If not improving, reduce learning rate
            lr *= 0.5
            if lr < 1e-10:
                break
            # Dampen momentum
            v *= 0.5
            
        # Convergence check
        if np.linalg.norm(g) < tol:
            break
            
    return best_x, best_e

def construct():
    t_start = time.time()
    limit = 115.0 # Safety margin before 120s
    
    # Initial configuration
    x = fibonacci_spiral()
    x, e = local_optimize(x, max_iters=5000)
    best_x, best_e = x, e
    
    # Adaptive noise scale for perturbations
    noise_scale = 0.5
    
    while time.time() - t_start < limit:
        # Perturb the best solution
        x_pert = best_x.copy()
        
        # Strategy: Mix of small noise and random kicks
        # 10% chance to do a "big kick" (replace random points)
        if np.random.rand() < 0.1:
            # Select 20 random points to replace
            num_kicks = 20
            idx = np.random.choice(N, num_kicks, replace=False)
            random_pts = np.random.randn(num_kicks, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
            x_pert[idx] = random_pts
        else:
            # Add Gaussian noise
            noise = np.random.randn(N, 3) * noise_scale
            x_pert += noise
            x_pert = x_pert / np.linalg.norm(x_pert, axis=1, keepdims=True)
        
        # Optimize the perturbed configuration
        x_opt, e_opt = local_optimize(x_pert, max_iters=2000)
        
        if e_opt < best_e:
            best_x, best_e = x_opt, e_opt
            # Decay noise scale on success
            noise_scale *= 0.95
            if noise_scale < 0.01:
                noise_scale = 0.01
        else:
            # Increase noise scale on failure
            noise_scale *= 1.05
            if noise_scale > 2.0:
                noise_scale = 2.0
                
    # Final check for validity
    # Ensure points are normalized
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x
