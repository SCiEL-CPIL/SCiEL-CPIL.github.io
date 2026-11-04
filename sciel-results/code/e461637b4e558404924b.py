import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# Pre-compute indices for all triangles i < j < k
# This avoids regenerating indices in every scoring call
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRI_INDICES = _get_triangle_indices(N)

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _neg_score(points_flat):
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(n1, n2, r, theta):
    """
    Generate two-ring configuration.
    n1 points on unit circle, n2 points on circle of radius r, rotated by theta.
    """
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + theta
    
    p1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r * np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    return np.vstack([p1, p2])

def _local_optimize(points, time_limit=15.0):
    """Nelder-Mead refinement on all 2N coordinates."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            s = -res.fun
            if s > best_score:
                best_score = s
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
        
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # 1. Try Regular 14-gon
    init_pts = _regular_polygon(N, 1.0)
    s, p = _local_optimize(init_pts, time_limit=10.0)
    if s > best_overall_score:
        best_overall_score = s
        best_overall_points = p
    
    # 2. Explore Two-Ring Configurations (7 inner, 7 outer)
    # This is a strong structural guess for N=14.
    # Parameters: r (radius of inner ring, 0 < r < 1), theta (rotation offset)
    
    # Coarse grid search over r and theta to find a good basin
    r_vals = np.linspace(0.1, 0.9, 9)
    theta_vals = np.linspace(0, np.pi/7, 8) # Symmetry period is pi/7 for heptagons
    
    best_struct_score = 0.0
    best_struct_pts = None
    
    for r in r_vals:
        if time.time() > deadline - 60:
            break
        for theta in theta_vals:
            pts = _two_ring_config(7, 7, r, theta)
            # Quick score to filter out bad candidates before local opt
            s_quick = _score(pts)
            if s_quick > best_struct_score:
                best_struct_score = s_quick
                best_struct_pts = pts
    
    # Refine the best structured candidate with local optimization
    if best_struct_pts is not None:
        s, p = _local_optimize(best_struct_pts, time_limit=20.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
            
    # 3. Try other ring splits? e.g., 6 and 8? 
    # Usually symmetric splits are better for convex hulls, but let's try 6/8 briefly if time permits
    if time.time() < deadline - 30:
        # 6 points outer, 8 points inner? Or vice versa?
        # Let's try 8 outer, 6 inner
        for r in [0.3, 0.5, 0.7]:
            for theta in [0.0, np.pi/8, np.pi/12]:
                pts = _two_ring_config(8, 6, r, theta)
                s, p = _local_optimize(pts, time_limit=5.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
                if time.time() > deadline - 20:
                    break
            if time.time() > deadline - 20:
                break

    # 4. Final Polish on the global best
    if time.time() < deadline - 5 and best_overall_points is not None:
        s, p = _local_optimize(best_overall_points, time_limit=10.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p

    if best_overall_points is None:
        best_overall_points = init_pts

    return best_overall_points
