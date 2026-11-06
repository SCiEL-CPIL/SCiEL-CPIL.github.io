import numpy as np
import time
import math

N = 100

def _energy(points):
    # points: (N, 3)
    # dists: (N, N)
    dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, 1.0) # Avoid log(0)
    # We only sum i < j
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    return -np.sum(np.log(dists[iu]))

def _gradient(points):
    # points: (N, 3)
    # diff: (N, N, 3) where diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    # grad[i] = - sum_j (diff[i,j] / dists[i,j]^2)
    # Note: The force on i from j is proportional to (x_i - x_j) / d^2.
    # Energy E = - sum log(d). dE/dx_i = - sum (1/d) * d(d)/dx_i = - sum (1/d) * (x_i - x_j)/d = - sum (x_i - x_j)/d^2.
    # So Gradient of Energy is - sum (x_i - x_j)/d^2.
    # We want to minimize E, so we move in direction -Gradient.
    # Direction = sum (x_i - x_j)/d^2.
    
    # We compute G[i] = sum_j (x_i - x_j)/d_ij^2
    # This is the negative of the energy gradient.
    # So we move in direction G[i].
    
    inv_d2 = np.zeros_like(dists)
    inv_d2[dists > 0] = 1.0 / (dists[dists > 0] ** 2)
    # Zero out diagonal
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_neg[i] = sum_j inv_d2[i,j] * diff[i,j]
    # diff is (N, N, 3), inv_d2 is (N, N)
    # Use einsum: 'ij,ijk->ik'
    grad_neg = np.einsum('ij,ijk->ik', inv_d2, diff)
    return grad_neg

def _optimize_single(points, time_limit=10.0):
    # points: (N, 3) array
    points = points.copy()
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_energy = _energy(points)
    best_points = points.copy()
    
    start_time = time.time()
    alpha = 0.01
    min_alpha = 1e-8
    max_alpha = 1.0
    
    for iter in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        grad = _gradient(points)
        
        # Project gradient to tangent space
        # g_tang = g - (g . x) x
        dot = np.sum(points * grad, axis=1, keepdims=True)
        g_tang = grad - dot * points
        
        # Check convergence
        norm_g = np.linalg.norm(g_tang)
        if norm_g < 1e-10:
            break
            
        # Line search
        for _ in range(20):
            new_points = points + alpha * g_tang
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            new_energy = _energy(new_points)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
                points = new_points
                alpha = min(alpha * 1.5, max_alpha)
                break
            else:
                alpha *= 0.5
                if alpha < min_alpha:
                    break
        else:
            # If we couldn't find a step, try smaller alpha or break
            if alpha < min_alpha:
                break
                
    return best_points, best_energy

def construct():
    # Deadline guard
    total_deadline = 120.0
    start_global = time.time()
    
    best_points = None
    best_energy = np.inf
    
    # Strategy: Multiple restarts
    # 1. Fibonacci spiral
    # 2. Random uniform
    # 3. Random with repulsion
    
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    def random_uniform():
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # Run optimization on several initializations
    # We have ~120s. Let's allocate time per restart.
    # If we do 10 restarts, we have 12s each.
    
    inits = [fibonacci_spiral()]
    for _ in range(9):
        inits.append(random_uniform())
        
    for i, init_pts in enumerate(inits):
        if time.time() - start_global > total_deadline - 5: # 5s safety margin
            break
            
        # Allocate remaining time to this optimization
        remaining = total_deadline - 5 - (time.time() - start_global)
        if remaining < 1.0:
            break
            
        # Limit single optimization to remaining time
        pts, energy = _optimize_single(init_pts, time_limit=remaining)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts
            
    # Final polish on the best point found with a tight tolerance
    # We might have time left
    if time.time() - start_global < total_deadline - 2:
        remaining = total_deadline - 2 - (time.time() - start_global)
        if remaining > 0.1:
            best_points, best_energy = _optimize_single(best_points, time_limit=remaining)
            
    return best_points
