import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

N = 13
# Pre-compute triangle indices for vectorized area calculation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute areas of all triangles formed by pairs of points."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute area of the convex hull of the points."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_score(points, temperature=0.1):
    """
    Smooth surrogate: soft-min of triangle areas divided by hull area.
    Uses log-sum-exp for numerical stability.
    """
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return -1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min: -log(sum(exp(-areas/T))) / T
    # We want to maximize soft_min, so minimize -soft_min / hull_area
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -temperature * log_sum
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, max_iter=2000, tol=1e-10):
    """
    Coordinate descent on individual point coordinates.
    For each point, optimize its (x, y) while keeping others fixed.
    This is effective for the min-triangle-area objective.
    """
    points = points.copy()
    n = len(points)
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            # Optimize point i's coordinates
            best_x = points[i][0]
            best_y = points[i][1]
            best_score = _score(points)
            
            # Try small perturbations in x and y
            for dx in [-0.01, 0.01, -0.001, 0.001, -0.0001, 0.0001]:
                for dy in [-0.01, 0.01, -0.001, 0.001, -0.0001, 0.0001]:
                    test_pts = points.copy()
                    test_pts[i][0] = best_x + dx
                    test_pts[i][1] = best_y + dy
                    s = _score(test_pts)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_x = test_pts[i][0]
                        best_y = test_pts[i][1]
                        improved = True
            
            points[i][0] = best_x
            points[i][1] = best_y
        
        if not improved:
            break
    
    return points

def _local_polish(points, deadline):
    """
    Multi-stage local polishing:
    1. Smooth objective with Powell
    2. Exact objective with Nelder-Mead
    3. Coordinate descent
    """
    current = points.copy()
    
    # Stage 1: Smooth objective with Powell
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _soft_min_score(x.reshape(N, 2), temperature=0.1),
                current.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            new_pts = res.x.reshape(N, 2)
            if _score(new_pts) > _score(current):
                current = new_pts
        except Exception:
            pass
    
    # Stage 2: Exact objective with Nelder-Mead
    if time.time() < deadline - 3:
        try:
            res_nm = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            new_pts = res_nm.x.reshape(N, 2)
            if _score(new_pts) > _score(current):
                current = new_pts
        except Exception:
            pass
    
    # Stage 3: Coordinate descent
    if time.time() < deadline - 1:
        current = _coordinate_descent_polish(current, max_iter=500, tol=1e-12)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate diverse starting points
    start_points = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(regular_13)
    
    # 2. Two-ring configurations with various parameters
    for n_out in [4, 5, 6, 7, 8, 9]:
        n_in = N - n_out
        if n_in < 1:
            continue
        for r_in in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), np.pi / (3 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                start_points.append(pts)
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(20):
        start_points.append(np.random.uniform(-1, 1, (N, 2)))
    
    # 4. Some structured perturbations of the regular polygon
    for pert in [0.05, 0.1, 0.2]:
        for angle_shift in [0, np.pi/N, 2*np.pi/N]:
            angles = np.linspace(angle_shift, 2 * np.pi + angle_shift, N, endpoint=False)
            radii = 1.0 + pert * np.sin(2 * angles)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            start_points.append(pts)
    
    # Polish each starting point
    for i, init_pts in enumerate(start_points):
        if time.time() > deadline - 10:
            break
        
        # Quick score check
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()
        
        # Local polish
        polished = _local_polish(init_pts, deadline)
        _update_best(polished)
    
    # Final global polish on the best point found
    if best_points is not None and time.time() < deadline - 2:
        final = _local_polish(best_points, deadline)
        _update_best(final)
    
    # Fallback
    if best_points is None:
        return np.random.rand(N, 2)
    
    return best_points
