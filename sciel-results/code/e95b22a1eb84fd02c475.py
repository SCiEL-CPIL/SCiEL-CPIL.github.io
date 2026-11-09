import numpy as np
import time
import math

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy_and_grad(pts):
        # pts: (N, 3)
        # Compute pairwise distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero (though points should be distinct)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # We compute sum over all i!=j and divide by 2
        # mask diagonal
        mask = ~np.eye(N, dtype=bool)
        log_dist = np.log(dist)
        # sum over i<j is 0.5 * sum_{i!=j} log(dist_ij)
        total_log_dist = np.sum(log_dist[mask])
        energy = -0.5 * total_log_dist
        
        # Gradient:
        # grad_i = sum_{j != i} (pts_j - pts_i) / dist_ij^2
        # Note: d/dx_i (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2 = (xj-xi)/|xi-xj|^2
        # Let's compute G such that G[i] = sum_{j != i} (pts[j] - pts[i]) / dist_sq[i,j]
        # diff[i,j] = pts[i] - pts[j]
        # We want (pts[j] - pts[i]) / dist_sq[i,j] = -diff[i,j] / dist_sq[i,j]
        
        # inv_dist_sq = 1 / dist_sq
        # Set diagonal to 0
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=dist_sq > 1e-12)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad[i] = sum_j (-diff[i,j]) * inv_dist_sq[i,j]
        # diff is (N, N, 3)
        # term[i, j, k] = -diff[i, j, k] * inv_dist_sq[i, j]
        term = -diff * inv_dist_sq[:, :, np.newaxis]
        grad = np.sum(term, axis=1)
        
        return energy, grad

    # Optimization loop
    alpha = 0.01
    best_energy, best_points = energy_and_grad(points)
    best_energy_val = best_energy
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        e, g = energy_and_grad(points)
        
        # Check convergence
        if abs(e - best_energy_val) < 1e-12:
            break
            
        # Backtracking line search
        # We want to minimize e(pts - alpha * g)
        # Since we project to sphere, we just check energy after projection
        new_alpha = alpha
        found = False
        for _ in range(20):
            new_pts = points - new_alpha * g
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_e, _ = energy_and_grad(new_pts)
            if new_e < e:
                found = True
                break
            new_alpha *= 0.5
            
        if not found:
            # If we can't decrease, maybe we are stuck or alpha is too small
            # Try smaller alpha or break
            if new_alpha < 1e-8:
                break
            # Use the best one found in the loop (which is new_e from last iter)
            # Actually, if no decrease, we might just break or keep best
            # Let's just break if we can't make progress
            break
            
        points = new_pts
        best_energy_val = new_e
        best_points = points.copy()
        alpha = new_alpha * 1.5 # Try to increase step size if it worked
        
    return best_points

# EVOLVE-BLOCK-END
