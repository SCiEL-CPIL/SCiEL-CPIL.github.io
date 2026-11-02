import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

def points_to_bary(pts):
    """Convert cartesian points to barycentric coordinates (u, v) where w = 1-u-v.
    pts: (N, 2)
    Returns: (N, 2) array of (u, v) such that p = u*V0 + v*V1 + w*V2
    Note: Standard barycentric: p = u*V0 + v*V1 + w*V2 with u+v+w=1.
    Let's use the basis:
    p = V0 + u*(V1-V0) + v*(V2-V0)
    u, v >= 0, u+v <= 1.
    """
    # p - V0 = u*(V1-V0) + v*(V2-V0)
    # [1, 0]   = u*[1, 0] + v*[0.5, sqrt(3)/2]
    # [x, y]   = u*[1, 0] + v*[0.5, sqrt(3)/2]
    # y = v * sqrt(3)/2  => v = 2*y/sqrt(3)
    # x = u + 0.5*v      => u = x - 0.5*v
    v = 2.0 * pts[:, 1] / np.sqrt(3.0)
    u = pts[:, 0] - 0.5 * v
    return np.column_stack((u, v))

def bary_to_points(uv):
    """Convert barycentric (u, v) to cartesian.
    uv: (N, 2)
    """
    u = uv[:, 0]
    v = uv[:, 1]
    x = u + 0.5 * v
    y = (np.sqrt(3.0) / 2.0) * v
    return np.column_stack((x, y))

def is_inside_triangle(pts):
    """Check if points are inside the triangle with small margin."""
    uv = points_to_bary(pts)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    margin = 1e-5
    return np.all(u >= -margin) and np.all(v >= -margin) and np.all(w >= -margin)

def min_triangle_area(pts):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = pts[i]
                p2 = pts[j]
                p3 = pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def objective_flat(flat_pts):
    """Objective function for optimizer. Minimizes negative min area."""
    pts = flat_pts.reshape(-1, 2)
    # Penalty if outside
    if not is_inside_triangle(pts):
        # Calculate distance to boundary to create a smooth penalty
        uv = points_to_bary(pts)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        dist = np.min(np.column_stack((u, v, w)), axis=1)
        penalty = np.sum(np.maximum(0, -dist)) * 100.0
        return 10.0 + penalty
    
    area = min_triangle_area(pts)
    return -area

def nudge_minimal_triangles(pts, rng, num_steps=100, step_size=0.01):
    """
    Identify the triangle(s) with minimum area and perturb the points involved.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_area = min_triangle_area(best_pts)
    
    # Find indices of the minimal triangle
    min_area = np.inf
    min_idx = None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = best_pts[i]
                p2 = best_pts[j]
                p3 = best_pts[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    
    if min_idx is None:
        return best_pts
        
    i, j, k = min_idx
    
    for _ in range(num_steps):
        # Try perturbing one of the points
        pt_idx = rng.choice([i, j, k])
        original_pt = best_pts[pt_idx].copy()
        
        # Generate a random direction
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        
        # Try a few step sizes
        found_improvement = False
        for factor in [step_size, step_size*0.5, step_size*0.1, -step_size, -step_size*0.5]:
            new_pt = original_pt + factor * direction
            test_pts = best_pts.copy()
            test_pts[pt_idx] = new_pt
            
            if is_inside_triangle(test_pts):
                new_area = min_triangle_area(test_pts)
                if new_area > best_area:
                    best_pts = test_pts
                    best_area = new_area
                    found_improvement = True
                    break
        
        if not found_improvement:
            # Try moving away from the centroid of the minimal triangle?
            # Or just random jitter
            pass
            
    return best_pts

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Initial candidates
    candidates = []
    
    # 1. Symmetric / Grid based initializations
    # Try a few structured layouts
    # Layout 1: 3x3 grid in barycentric, plus center? No, 11 points.
    # Let's try random restarts with a good seed.
    
    rng = np.random.default_rng(42)
    
    # Generate a few diverse starting points
    for init_id in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random points inside triangle
        # Using barycentric coordinates
        u = rng.random(N)
        v = rng.random(N)
        # Normalize to be inside
        # This simple method doesn't ensure uniform distribution, but it's okay for init
        # Better: generate in unit square, reject if u+v>1
        while True:
            u = rng.random(N)
            v = rng.random(N)
            if np.all(u + v <= 1.0):
                break
        
        pts = bary_to_points(np.column_stack((u, v)))
        
        # Local optimization
        flat_pts = pts.flatten()
        
        # Nelder-Mead
        try:
            res = minimize(objective_flat, flat_pts, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500})
            if res.success or res.fun < 0:
                best_pts = res.x.reshape(-1, 2)
                # Nudge phase
                if time.time() - start_time < time_limit - 10:
                    best_pts = nudge_minimal_triangles(best_pts, rng, num_steps=50)
                    # Re-optimize after nudge
                    res2 = minimize(objective_flat, best_pts.flatten(), method='Nelder-Mead', 
                                    options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200})
                    if res2.fun < res.fun:
                        best_pts = res2.x.reshape(-1, 2)
                
                area = min_triangle_area(best_pts)
                candidates.append((area, best_pts))
        except Exception:
            pass

    # 2. Random restarts with remaining time
    while time.time() - start_time < time_limit - 5:
        u = rng.random(N)
        v = rng.random(N)
        while np.any(u + v > 1.0):
            u = rng.random(N)
            v = rng.random(N)
            
        pts = bary_to_points(np.column_stack((u, v)))
        flat_pts = pts.flatten()
        
        try:
            res = minimize(objective_flat, flat_pts, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 300})
            best_pts = res.x.reshape(-1, 2)
            
            # Quick nudge
            if time.time() - start_time < time_limit - 10:
                best_pts = nudge_minimal_triangles(best_pts, rng, num_steps=20)
            
            area = min_triangle_area(best_pts)
            candidates.append((area, best_pts))
        except Exception:
            pass

    if not candidates:
        # Fallback
        u = np.random.random(N)
        v = np.random.random(N)
        while np.any(u + v > 1.0):
            u = np.random.random(N)
            v = np.random.random(N)
        pts = bary_to_points(np.column_stack((u, v)))
        return pts

    # Select best
    best_area, best_pts = max(candidates, key=lambda x: x[0])
    
    # Final check
    if not is_inside_triangle(best_pts):
        # Clip to boundary? Or just return the best valid one from candidates
        # The penalty should have prevented this, but just in case
        for area, pts in sorted(candidates, reverse=True, key=lambda x: x[0]):
            if is_inside_triangle(pts):
                return pts
        # If all invalid, return a valid random set
        u = np.random.random(N)
        v = np.random.random(N)
        while np.any(u + v > 1.0):
            u = np.random.random(N)
            v = np.random.random(N)
        return bary_to_points(np.column_stack((u, v)))

    return best_pts
