"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TIME_LIMIT = 115  # Safety margin for 120s limit

def _min_area_raw(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # We need area for all (i, j, k) with i < j < k
    # Using cross product: 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # To avoid O(N^3) python loop overhead, we can use broadcasting or just a tight loop.
    # For N=11, N^3 is ~1331, so a tight python loop is fast enough.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a

def _is_inside_triangle(p):
    """Check if a point (x, y) is inside the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates or half-plane tests
    # 1. y >= 0
    if p[1] < 0:
        return False
    # 2. y <= sqrt(3)/2
    if p[1] > S3H + 1e-9:
        return False
    # 3. Left edge: y <= sqrt(3) * x
    if p[1] > np.sqrt(3) * p[0] + 1e-9:
        return False
    # 4. Right edge: y <= -sqrt(3) * (x - 1)
    if p[1] > -np.sqrt(3) * (p[0] - 1) + 1e-9:
        return False
    return True

def _penalized_min_area_flat(x):
    """
    Objective function for minimization (we want to maximize min area, so minimize -min_area).
    x is a flat array of shape (2*N,).
    """
    points = x.reshape(N, 2)
    
    # Check validity and apply penalty if outside
    # We use a soft penalty to allow the optimizer to move points back inside
    # instead of hard constraints which can stall Nelder-Mead.
    penalty = 0.0
    for i in range(N):
        if not _is_inside_triangle(points[i]):
            # Distance to boundary approximation or just a large penalty
            # Simple penalty: sum of violations
            # y < 0
            if points[i, 1] < 0:
                penalty += (0 - points[i, 1])**2
            # y > S3H
            if points[i, 1] > S3H:
                penalty += (points[i, 1] - S3H)**2
            # y > sqrt(3)*x
            if points[i, 1] > np.sqrt(3) * points[i, 0]:
                penalty += (points[i, 1] - np.sqrt(3) * points[i, 0])**2
            # y > -sqrt(3)*(x-1)
            if points[i, 1] > -np.sqrt(3) * (points[i, 0] - 1):
                penalty += (points[i, 1] - (-np.sqrt(3) * (points[i, 0] - 1)))**2
            
    min_a = _min_area_raw(points)
    
    # If any point is way outside, return a large value
    if penalty > 1e-4:
        return 1.0 + penalty * 1000
    
    return -min_a + penalty * 1000

def _local_polish(points, time_budget):
    """Run Nelder-Mead starting from points."""
    x0 = points.flatten()
    
    # Nelder-Mead options
    options = {
        'maxiter': 2000,
        'fatol': 1e-12,
        'xatol': 1e-12
    }
    
    try:
        res = minimize(_penalized_min_area_flat, x0, method='Nelder-Mead', options=options)
        if res.success and _penalized_min_area_flat(res.x) < _penalized_min_area_flat(x0):
            # Check if result is valid (inside triangle)
            p_res = res.x.reshape(N, 2)
            if all(_is_inside_triangle(p) for p in p_res):
                return p_res
    except Exception:
        pass
        
    return points

def _construct_grid_initial():
    """
    Construct a high-quality initial guess using a structured grid.
    We want 11 points.
    A 3x3 grid has 9 points. Adding 2 more is tricky.
    Alternatively, use a 4x4 grid subset or a specific pattern.
    
    Let's try a "quasi-random" but structured approach or a specific known good pattern.
    Actually, a simple stratified sampling or a grid with jitter is good.
    
    Let's try a 4x4 grid (16 points) and select 11? No, that might be dense.
    Let's try a 3x3 grid (9 points) plus center and one other?
    
    Better: Use a 5x5 grid (25 points) and select the 11 that maximize min distance?
    Or just use a fixed good starting point.
    
    Let's use a "hexagonal" packing approximation or a grid.
    Grid 3x3:
    (0,0), (1/3,0), (2/3,0)
    (0, h), (1/3,h), (2/3,h)
    (0,2h), (1/3,2h), (2/3,2h)
    where h = S3H/3.
    
    This puts points on the boundary. We should nudge them in.
    Let's create a 3x3 grid slightly inside.
    """
    points = []
    # 3x3 grid
    # x coords: 0.1, 0.5, 0.9 ? No, triangle shape varies.
    # Let's use barycentric coords for grid.
    # Barycentric (u, v, w) with u+v+w=1.
    # Grid steps: 0.1, 0.5, 0.9?
    
    # Let's just use a deterministic set of 11 points that are well distributed.
    # 1 center, 6 in a hexagon, 4 in corners?
    
    # Simpler: Stratified random with a fixed seed, but structured.
    # We will generate a 5x5 grid of barycentric coordinates and pick 11.
    
    # Barycentric grid 5x5 (indices 0..4)
    # u = i/4, v = j/4, w = 1 - u - v
    # Valid if u>=0, v>=0, w>=0
    
    candidates = []
    for i in range(5):
        for j in range(5):
            u = i / 4.0
            v = j / 4.0
            w = 1.0 - u - v
            if u >= 0 and v >= 0 and w >= 0:
                # Convert to cartesian
                # p = u*A + v*B + w*C
                # A=(0,0), B=(1,0), C=(0.5, S3H)
                x = u*0 + v*1 + w*0.5
                y = u*0 + v*0 + w*S3H
                candidates.append([x, y])
                
    # We have 15 valid points in 5x5 grid (triangular number T5 = 15).
    # We need 11.
    # Which 11 to keep? The ones that maximize min distance?
    # Greedy selection:
    selected = []
    # Start with center
    center = [0.5, S3H/3] # Centroid
    selected.append(center)
    
    remaining = [p for p in candidates if not np.allclose(p, center)]
    
    for _ in range(N - 1):
        best_p = None
        best_min_dist = -1
        for p in remaining:
            min_d = min(np.linalg.norm(p - q) for q in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_p = p
        selected.append(best_p)
        remaining.remove(best_p)
        
    return np.array(selected)

def construct():
    start_time = time.time()
    
    # 1. Generate initial guess
    points = _construct_grid_initial()
    
    # 2. Local polish
    points = _local_polish(points, TIME_LIMIT - (time.time() - start_time))
    
    # 3. Multi-start with random perturbations if time remains
    best_points = points
    best_score = _min_area_raw(best_points)
    
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < TIME_LIMIT - 5:
        # Perturb the best points found so far
        noise_scale = 0.05
        perturbed = best_points + rng.randn(N, 2) * noise_scale
        
        # Project back to triangle if needed? 
        # The penalty in _penalized_min_area_flat handles it, but Nelder-Mead might struggle if too far out.
        # Let's clamp or project.
        # Simple projection: if outside, move towards centroid.
        for i in range(N):
            if not _is_inside_triangle(perturbed[i]):
                # Move towards centroid (0.5, S3H/3)
                centroid = np.array([0.5, S3H/3])
                perturbed[i] = centroid + 0.5 * (perturbed[i] - centroid)
                # If still outside, just use centroid
                if not _is_inside_triangle(perturbed[i]):
                    perturbed[i] = centroid
        
        # Polish this perturbed version
        polished = _local_polish(perturbed, TIME_LIMIT - (time.time() - start_time))
        
        score = _min_area_raw(polished)
        if score > best_score:
            best_score = score
            best_points = polished
            
    return best_points
# EVOLVE-BLOCK-END
