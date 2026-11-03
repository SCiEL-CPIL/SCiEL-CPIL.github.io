"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
import random
import sys

def _compute_min_tri_area_norm(pts):
    """
    Compute the minimum area of any triangle formed by pts,
    normalized by the area of the convex hull of pts.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-9:
            return -1e9
    except Exception:
        return -1e9

    # Compute all triangle areas
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                  p2[0] * (p3[1] - p1[1]) + 
                                  p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _objective(params):
    pts = params.reshape(N, 2)
    val = _compute_min_tri_area_norm(pts)
    return -val

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s kill limit
    
    # Strategy:
    # 1. Start with a regular polygon (good baseline).
    # 2. Use L-BFGS-B to optimize the max-min objective.
    # 3. Use multiple restarts with different initializations (Halton, Random, Perturbed Regular)
    #    to escape local minima. The "min" function is non-smooth, so restarts are crucial.
    
    best_score = -1e9
    best_pts = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    p0 = np.column_stack((x0, y0)).flatten()
    
    # List of initial points to try
    initializations = [p0]
    
    # Halton sequence initialization (low-discrepancy)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    halton_arr = np.array(halton_pts)
    # Center and scale
    halton_arr -= np.mean(halton_arr, axis=0)
    halton_arr *= 1.0
    initializations.append(halton_arr.flatten())
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        rand_pts = rng.random((N, 2)) * 2 - 1
        rand_pts -= np.mean(rand_pts, axis=0)
        initializations.append(rand_pts.flatten())
    
    # Perturbations of regular polygon
    for _ in range(3):
        noise = rng.normal(0, 0.2, size=N * 2)
        perturbed = p0 + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        initializations.append(perturbed.flatten())
    
    # Optimize each initialization
    for init in initializations:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'maxfun': 50000, 'disp': False}
            )
            
            score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_pts = res.x.copy()
                
        except Exception as e:
            pass
    
    # If we have time, do a few more local refinements from the best solution found
    if best_pts is not None:
        while time.time() - start_time < time_limit - 5.0:
            rng = np.random.default_rng()
            noise = rng.normal(0, 0.05, size=N * 2)
            p_perturbed = best_pts + noise
            p_perturbed = p_perturbed.reshape(N, 2)
            p_perturbed -= np.mean(p_perturbed, axis=0)
            
            try:
                res = minimize(
                    _objective, 
                    p_perturbed.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'maxfun': 20000, 'disp': False}
                )
                score = _compute_min_tri_area_norm(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.copy()
            except Exception:
                pass
            # Small pause to avoid CPU saturation issues? No, just loop.
            # To prevent infinite tight loop if no improvement, we can just break after a few
            # or rely on time. The prompt says "Keep the function names...".
            # Let's add a counter to prevent infinite loop if time check fails (unlikely).
            # Actually, just breaking after a few refinements is safer if time is tight.
            # But the while condition handles it.
    
    if best_pts is None:
        # Fallback to regular polygon
        return np.column_stack((x0, y0))
        
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
