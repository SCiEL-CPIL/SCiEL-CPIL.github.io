"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Coordinate Descent with Smooth Soft-Min Objective.
We iteratively optimize each point individually while fixing the others.
This decomposes the high-dimensional search into 11 sequential 2D optimizations,
using L-BFGS-B on a log-sum-exp approximation of the minimum area.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _soft_min_objective_single(x_flat, fixed_pts, idx, temperature=1.0):
    """
    Objective for optimizing a single point x_flat (2D) given fixed other points.
    Returns -soft_min_area + penalty.
    """
    # Reconstruct full points array
    pts = fixed_pts.copy()
    pts[idx] = x_flat
    
    # Penalty for being outside the triangle
    penalty = 0.0
    x, y = x_flat
    # Distance to boundaries
    d1 = y # bottom
    d2 = np.sqrt(3) * x - y # left
    d3 = np.sqrt(3) * (1 - x) - y # right
    
    if d1 < 0: penalty += d1**2
    if d2 < 0: penalty += d2**2
    if d3 < 0: penalty += d3**2
    
    # Compute all triangle areas involving the moving point? 
    # Actually, the minimum area is over ALL triples. 
    # Moving one point affects all triples containing it.
    # Triples not containing it remain constant.
    # For efficiency, we can pre-compute the min area of triples NOT containing idx.
    # But for simplicity and correctness in this function, we compute all.
    # Since N=11, total triples = 165. This is fast enough for 2D optimization.
    
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                
    areas = np.array(areas)
    
    # Soft minimum approximation
    b = 1.0 / temperature
    areas_clipped = np.clip(areas, 1e-6, None)
    exp_terms = np.exp(-b * areas_clipped)
    soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
    
    obj = -soft_min_val
    obj += 100.0 * penalty
    
    return obj

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points."""
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Coordinate Descent with local L-BFGS-B optimization for each point.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # Temperature for soft min: lower is sharper (closer to hard min)
    # Start with a moderate temperature to allow smooth gradients, then refine
    temp = 0.05
    
    # Multiple restarts to avoid local minima
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Initialize
        current_pts = _initial_guess(rng)
        current_pts = _project_to_triangle(current_pts)
        
        # Evaluate initial
        if _check_inside(current_pts):
            area = _min_triangle_area(current_pts)
            if area > best_score:
                best_score = area
                best_pts = current_pts.copy()
        
        # Coordinate Descent Loop
        # We iterate through points multiple times (sweeps)
        max_sweeps = 50
        
        for sweep in range(max_sweeps):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Decrease temperature slowly to sharpen the minimum
            # But be careful not to make it too non-smooth for L-BFGS
            # We'll keep it relatively stable but decrease slightly
            temp = max(0.01, 0.05 * (0.95 ** sweep))
            
            for idx in range(N):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                fixed_pts = current_pts.copy()
                initial_x = current_pts[idx]
                
                # Define objective for this specific point
                def obj_func(x):
                    return _soft_min_objective_single(x, fixed_pts, idx, temperature=temp)
                
                # Use L-BFGS-B
                # Bounds: The triangle constraints are handled by projection or penalty.
                # L-BFGS-B works better with explicit bounds if possible, but the triangle
                # is non-axis-aligned. We rely on the penalty and projection.
                # However, we can provide loose bounds [0, 1] for x and [0, S3H] for y
                # to keep the optimizer in the general vicinity.
                bounds = [(0.0, 1.0), (0.0, S3H)]
                
                try:
                    result = minimize(
                        obj_func,
                        initial_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 20, 'maxfun': 50} # Keep it fast per point
                    )
                    
                    new_x = result.x
                    # Project to ensure strict feasibility
                    new_x = np.array([new_x[0], new_x[1]])
                    # Simple projection logic for single point
                    x, y = new_x
                    if y < 0: y = 0
                    max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
                    if y > max_y: y = max_y
                    if x < 0: x = 0
                    if x > 1: x = 1
                    
                    current_pts[idx] = [x, y]
                    
                    # Update best if improved
                    if _check_inside(current_pts):
                        area = _min_triangle_area(current_pts)
                        if area > best_score:
                            best_score = area
                            best_pts = current_pts.copy()
                            
                except Exception:
                    pass
    
    # Final polish on the best found configuration
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        # Run a few more sweeps on the best solution with a lower temperature
        final_pts = best_pts.copy()
        final_temp = 0.005
        
        for sweep in range(20):
            if time.time() - start_time > DEADLINE - 2:
                break
            for idx in range(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                fixed_pts = final_pts.copy()
                initial_x = final_pts[idx]
                
                def obj_func_final(x):
                    return _soft_min_objective_single(x, fixed_pts, idx, temperature=final_temp)
                
                bounds = [(0.0, 1.0), (0.0, S3H)]
                
                try:
                    result = minimize(
                        obj_func_final,
                        initial_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 30, 'maxfun': 80}
                    )
                    new_x = result.x
                    x, y = new_x
                    if y < 0: y = 0
                    max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
                    if y > max_y: y = max_y
                    if x < 0: x = 0
                    if x > 1: x = 1
                    
                    final_pts[idx] = [x, y]
                    
                    if _check_inside(final_pts):
                        area = _min_triangle_area(final_pts)
                        if area > best_score:
                            best_score = area
                            best_pts = final_pts.copy()
                except Exception:
                    pass

    if best_pts is None:
        # Fallback
        best_pts = _initial_guess(rng)
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts
