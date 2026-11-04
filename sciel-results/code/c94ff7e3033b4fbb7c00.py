import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Triangle area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min_triangle_area / hull_area. Return negative for minimization."""
    points = np.asarray(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return -min_area / hull_area

def _soft_min_triangle_area(points, beta=100.0):
    """Soft-min approximation of minimum triangle area for smooth optimization."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # Use log-sum-exp trick for stability
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) + (-beta * min_val)
    soft_min = -log_sum / beta
    return soft_min

def _smooth_objective(x, beta=100.0):
    """Smooth objective for L-BFGS-B: minimize -soft_min_area / hull_area."""
    points = x.reshape(-1, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    soft_min = _soft_min_triangle_area(points, beta)
    return -soft_min / hull_area

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular n-gon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

def _concentric_rings(n, inner_radius=0.4, outer_radius=1.0, n_inner=5):
    """Generate points on concentric rings."""
    points = []
    # Outer ring
    n_outer = n - n_inner
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    for a in angles_outer:
        points.append([outer_radius * np.cos(a), outer_radius * np.sin(a)])
    # Inner ring
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
    for a in angles_inner:
        points.append([inner_radius * np.cos(a), inner_radius * np.sin(a)])
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    # Generate initial configurations
    inits = []
    
    # 1. Regular 13-gon
    inits.append(_regular_polygon(N, radius=1.0))
    
    # 2. Concentric rings with various parameters
    for inner_r in [0.2, 0.3, 0.4, 0.5]:
        for n_inner in [3, 4, 5, 6]:
            if n_inner < N:
                inits.append(_concentric_rings(N, inner_radius=inner_r, outer_radius=1.0, n_inner=n_inner))
    
    # 3. Random points in disk
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng_local = np.random.RandomState(seed)
        angles = rng_local.uniform(0, 2*np.pi, N)
        radii = rng_local.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    # 4. Points with one center and rest on circle
    for seed in range(5):
        rng_local = np.random.RandomState(100 + seed)
        angles = rng_local.uniform(0, 2*np.pi, N-1)
        radii = rng_local.uniform(0.8, 1.0, N-1)
        outer = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center_pt = np.array([[0.0, 0.0]])
        pts = np.vstack([center_pt, outer])
        inits.append(pts)
    
    for init_pts in inits:
        if time.time() - start_time > deadline:
            break
        
        x0 = init_pts.flatten()
        
        # Try L-BFGS-B with soft-min
        try:
            result = minimize(
                _smooth_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success and time.time() - start_time < deadline:
                x_opt = result.x.reshape(-1, 2)
                score = -_score(x_opt)
                if score > best_score:
                    best_score = score
                    best_points = x_opt.copy()
        except Exception:
            pass
        
        # Polish with Nelder-Mead on exact objective
        if time.time() - start_time < deadline:
            try:
                start_x = best_points.flatten() if best_points is not None else x0
                result_nm = minimize(
                    _score,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if time.time() - start_time < deadline:
                    x_nm = result_nm.x.reshape(-1, 2)
                    score_nm = -_score(x_nm)
                    if score_nm > best_score:
                        best_score = score_nm
                        best_points = x_nm.copy()
            except Exception:
                pass
    
    # If no valid result found, return a reasonable default
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
