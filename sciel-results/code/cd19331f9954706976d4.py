"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Delaunay Triangulation-Based Active Set Optimization.
1. Use barycentric coordinates to naturally enforce the triangle constraint.
2. Leverage the theorem that the minimum area triangle in a point set must be a Delaunay triangle.
3. Iteratively:
   a. Compute Delaunay triangulation of the current point set.
   b. Identify the minimum area triangle in the Delaunay mesh (active set).
   c. Use a local optimization (SLSQP) to maximize the minimum area among the active set triangles.
   d. Re-compute Delaunay triangulation and repeat if topology changes.
4. Multi-start with diverse initializations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    if len(pts) < 3:
        return 0.0
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a = pts[i]
        b = pts[j]
        c = pts[k]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(cross)
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

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _get_delaunay_triangles(pts):
    """Get Delaunay triangles from a set of points."""
    try:
        tri = Delaunay(pts)
        return tri.simplices
    except Exception:
        return None

def _active_set_optimize(pts, max_iter=50, tol=1e-6):
    """
    Active set optimization using Delaunay triangulation.
    Iteratively:
    1. Compute Delaunay triangulation.
    2. Find minimum area triangle in the mesh.
    3. Optimize to increase the minimum area among active triangles.
    4. Repeat if topology changes.
    """
    pts = np.array(pts, dtype=float)
    best_pts = pts.copy()
    best_score = _min_triangle_area(pts)
    
    for iteration in range(max_iter):
        # Get Delaunay triangles
        tris = _get_delaunay_triangles(pts)
        if tris is None:
            break
        
        # Find minimum area triangle in Delaunay mesh
        min_area = float('inf')
        min_tris = []
        for tri in tris:
            i, j, k = tri
            a = pts[i]
            b = pts[j]
            c = pts[k]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            area = 0.5 * abs(cross)
            if area < min_area - 1e-9:
                min_area = area
                min_tris = [tri]
            elif abs(area - min_area) < 1e-9:
                min_tris.append(tri)
        
        if not min_tris:
            break
        
        # Define objective: maximize minimum area among active triangles
        # We'll use a soft-min approach for smoothness
        def objective(bary_flat):
            bary = bary_flat.reshape(N, 3)
            # Enforce barycentric constraints softly
            bary_sum = np.sum(bary, axis=1)
            bary = bary / bary_sum[:, None]
            pts_curr = _barycentric_to_cartesian(bary)
            
            # Compute areas of active triangles
            areas = []
            for tri in min_tris:
                i, j, k = tri
                a = pts_curr[i]
                b = pts_curr[j]
                c = pts_curr[k]
                cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                area = 0.5 * abs(cross)
                areas.append(area)
            
            # Soft minimum: -log(sum(exp(-areas/t))) * t
            t = 0.01
            log_sum = np.log(np.sum(np.exp(-np.array(areas) / t)))
            soft_min = -t * log_sum
            return -soft_min  # Minimize negative to maximize
        
        # Initial barycentric coordinates
        bary = _cartesian_to_bary(pts)
        bary_flat = bary.flatten()
        
        # Bounds
        bounds = [(0, 1)] * (N * 3)
        
        # Optimize
        try:
            result = minimize(objective, bary_flat, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.success:
                bary_opt = result.x.reshape(N, 3)
                bary_opt = bary_opt / np.sum(bary_opt, axis=1, keepdims=True)
                pts_new = _barycentric_to_cartesian(bary_opt)
                pts_new = _project_to_triangle(pts_new)
                
                if _check_inside(pts_new):
                    score_new = _min_triangle_area(pts_new)
                    if score_new > best_score:
                        best_score = score_new
                        best_pts = pts_new.copy()
                        pts = pts_new.copy()
                    else:
                        break
        except Exception:
            break
    
    return best_pts, best_score

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

def _greedy_init(rng):
    """Greedy initialization: place points to maximize minimum distance."""
    pts = []
    p0 = _random_point_in_triangle(rng)
    pts.append(np.array(p0))
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(300):
            cand = np.array(_random_point_in_triangle(rng))
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        pts.append(best_pos)
    
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Delaunay-based active set optimization with multi-start.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Multi-start initializations
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    initializations.append(p_rand)
    
    # 3. Structured: Vertices, Midpoints, Center, and additional points
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    p_struct = np.array(p_struct)
    p_struct = _project_to_triangle(p_struct)
    initializations.append(p_struct)
    
    # 4. Grid-like: 3x3 grid clipped to triangle
    p_grid = []
    for i in range(4):
        for j in range(4):
            x = i / 3.0
            y = j / 3.0 * S3H
            if y <= S3H * x and y <= S3H * (1 - x):
                p_grid.append([x, y])
    if len(p_grid) >= N:
        p_grid = np.array(p_grid[:N])
        p_grid = _project_to_triangle(p_grid)
        initializations.append(p_grid)
    
    # Run active set optimization from each initialization
    for s, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        try:
            pts_opt, score_opt = _active_set_optimize(init_pts)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, fall back to best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
    
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
    
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
