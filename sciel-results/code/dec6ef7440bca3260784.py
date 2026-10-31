"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def calc_areas(points):
    """Calculate areas of all triangles formed by the points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)

def dist_to_segment(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def penalty_outside(points):
    """Calculate penalty for points outside the triangle."""
    pen = 0.0
    h = np.sqrt(3) / 2
    for p in points:
        # Barycentric coordinates
        # p = u*A + v*B + w*C
        # x = v + 0.5w
        # y = h*w
        w = p[1] / h
        v = p[0] - 0.5 * w
        u = 1 - v - w
        
        if u < 0 or v < 0 or w < 0:
            # Determine which edge is violated
            d = 0.0
            if u < 0:
                d = max(d, dist_to_segment(p, B, C))
            if v < 0:
                d = max(d, dist_to_segment(p, A, C))
            if w < 0:
                d = max(d, dist_to_segment(p, A, B))
            # If multiple are negative, the point is near a vertex.
            # The max distance to the violated edges is a good proxy for "how far out".
            # Actually, just summing squared distances to violated edges is fine.
            pen += d * d
    return pen

def random_points_in_triangle(n, rng):
    """Generate n random points inside the triangle."""
    pts = []
    for _ in range(n):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        x = u + 0.5 * v
        y = (np.sqrt(3) / 2) * v
        pts.append([x, y])
    return np.array(pts)

def construct():
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110  # Keep 10s margin
    
    best_val = 0.0
    best_pts = None
    
    # We will run multiple optimization attempts
    num_starts = 50
    
    # Use a moderate temperature for soft-min
    tau = 20.0
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Initial guess
        # Mix of random and slightly structured (grid-like)
        if i % 5 == 0:
            # Grid-like initialization
            # Create a 4x4 grid, pick 11 points, scale to triangle
            # This is a heuristic, random is often better for non-convex
            pts = random_points_in_triangle(N, rng)
        else:
            pts = random_points_in_triangle(N, rng)
        
        # Perturb slightly to avoid symmetry traps
        pts += rng.uniform(-0.01, 0.01, pts.shape)
        
        # Ensure initial points are inside (project or regenerate)
        # For simplicity, we rely on the penalty to push them in if they start slightly out
        # But random generation ensures they are in.
        
        def objective(x):
            p = x.reshape(N, 2)
            areas = calc_areas(p)
            
            # Soft min of areas
            # To avoid overflow/underflow in exp, subtract max
            m = np.max(areas)
            if m < 1e-10:
                return 1e6  # Huge penalty if all areas are 0
            lse = m + np.log(np.sum(np.exp(areas - m)))
            soft_min_val = -lse / tau
            
            # Penalty for being outside
            pen = penalty_outside(p)
            
            # We want to maximize soft_min, so minimize -soft_min
            # Add penalty
            return -soft_min_val + 100.0 * pen
        
        # Optimize
        try:
            res = minimize(objective, pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                final_pts = res.x.reshape(N, 2)
                
                # Final evaluation
                final_areas = calc_areas(final_pts)
                min_area = np.min(final_areas)
                
                # Check if points are reasonably inside (penalty should be small)
                if penalty_outside(final_pts) < 1e-4:
                    if min_area > best_val:
                        best_val = min_area
                        best_pts = final_pts
        except Exception:
            pass

    # If no valid points found (should not happen), return random
    if best_pts is None:
        best_pts = random_points_in_triangle(N, rng)
        
    return best_pts

# EVOLVE-BLOCK-END
