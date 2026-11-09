"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def get_fibonacci_points(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize(pts, max_steps, time_limit):
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        step = 0.01
        momentum = 0.9
        v = np.zeros_like(pts)
        
        for step_idx in range(max_steps):
            if time.time() > time_limit:
                break
                
            grad = gradient(pts)
            
            v = momentum * v - step * grad
            pts_new = normalize(pts + v)
            
            e_new = log_energy(pts_new)
            
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
            
            pts = pts_new
            step *= 0.999
            if step < 1e-5:
                step = 1e-5
                
        return best_pts, best_e

    # Phase 1: Coarse optimization from Fibonacci start
    pts = get_fibonacci_points(N)
    pts, e1 = optimize(pts, 2000, start_time + 20.0)

    # Phase 2: Fine optimization
    # Reduce step size further for finer convergence
    step = 1e-4
    momentum = 0.8
    v = np.zeros_like(pts)
    best_pts = pts.copy()
    best_e = e1
    
    # Run until deadline
    max_steps = 100000
    for step_idx in range(max_steps):
        if time.time() > deadline - 5.0: # Leave 5s buffer for final checks/return
            break
            
        grad = gradient(pts)
        
        v = momentum * v - step * grad
        pts_new = normalize(pts + v)
        
        e_new = log_energy(pts_new)
        
        if e_new < best_e:
            best_e = e_new
            best_pts = pts_new.copy()
            
        pts = pts_new
        step *= 0.9995
        if step < 1e-6:
            step = 1e-6

    # Final cleanup
    best_pts = normalize(best_pts)
    
    # Check for invalid states (points too close)
    # If any pair is closer than 1e-9, perturb slightly
    # This is a safety check, unlikely to trigger with good optimization
    min_dist_sq = np.min(np.sum((best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :])**2, axis=2))
    # Actually min dist is off-diagonal.
    # Let's just ensure no zero norms or non-finite
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        return get_fibonacci_points(N)

    return best_pts

# EVOLVE-BLOCK-END
