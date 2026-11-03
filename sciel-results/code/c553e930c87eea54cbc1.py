import numpy as np
import math

N = 150

def construct():
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Gradient Descent on the Sphere
    # Energy E = sum_{i<j} log(1/|xi - xj|)
    # Gradient w.r.t xi: sum_{j != i} (xj - xi) / |xi - xj|^2
    # We want to minimize E, so we move in direction -Grad_E.
    # Force F_i = -Grad_E_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    # This is a repulsive force.
    
    iterations = 10000
    lr = 0.01
    decay = 0.9995
    
    best_points = points.copy()
    best_energy = np.inf
    
    for it in range(iterations):
        # Compute pairwise differences
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        # Compute squared distances
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero (diagonal)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute inverse squared distances
        inv_dist_sq = 1.0 / dist_sq
        
        # Set diagonal to 0 to ignore self-interaction
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Compute Force on each point i
        # F_i = sum_j (x_i - x_j) / |x_i - x_j|^2
        # diff[i, j] is x_i - x_j
        # We need sum_j diff[i, j] * inv_dist_sq[i, j]
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project forces to tangent space
        # F_tan = F - (F . x) x
        dot_prods = np.sum(forces * points, axis=1)
        forces_tangent = forces - dot_prods[:, np.newaxis] * points
        
        # Update points
        points = points + lr * forces_tangent
        
        # Normalize points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Decay learning rate
        lr *= decay
        
        # Periodically check energy
        if it % 100 == 0:
            # Calculate energy
            # Reuse dist_sq from above? No, points changed.
            # For efficiency in the check, we can just compute it.
            # dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
            # np.fill_diagonal(dists, 1.0)
            # triu = np.triu_indices(N, k=1)
            # d_pair = dists[triu]
            # E = -np.sum(np.log(d_pair))
            
            # Faster energy check:
            # We already have dist_sq logic, but it's for the current state before update?
            # No, we updated points. We need new distances.
            # Let's just compute it simply.
            dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
            np.fill_diagonal(dists, 1.0)
            iu = np.triu_indices(N, k=1)
            d_pair = dists[iu]
            E = -np.sum(np.log(d_pair))
            
            if E < best_energy:
                best_energy = E
                best_points = points.copy()
                
    return best_points
