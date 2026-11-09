"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_N_TRIS = len(_idx_i)

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    return np.min(_all_areas(points))

def _inside_penalty(points):
    """Calculate penalty for points outside the triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Constraints:
    1. y >= 0
    2. y <= sqrt(3)*x  => sqrt(3)*x - y >= 0
    3. y <= -sqrt(3)*(x-1) => sqrt(3)*(1-x) - y >= 0
    """
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = np.sqrt(3) * x - y
    d3 = np.sqrt(3) * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _smooth_min_area(points, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    areas = _all_areas(points)
    # Use log-sum-exp to approximate min
    min_a = np.min(areas)
    # For large beta, this approaches the min
    # logsumexp(-beta * areas) / (-beta) approximates min(areas)
    # But we want to minimize -min_area, so we maximize min_area
    # Let's use: -logsumexp(-beta * areas) / beta ≈ min(areas)
    # Actually for maximizing min, we can use:
    # min ≈ -logsumexp(-beta * areas) / beta
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    logsumexp_val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_min = -logsumexp_val / beta
    return smooth_min

def _objective_smooth(params, beta=50.0):
    """Objective for smooth global optimization: -smooth_min_area + penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 0.1:
        return 1e6 + pen
    smin = _smooth_min_area(points, beta)
    return -smin + 5.0 * pen

def _objective_exact(params):
    """Objective for exact local optimization: -min_area + penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 0.1:
        return 1e6 + pen
    area = _min_area(points)
    return -area + 5.0 * pen

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        # Clamp x to [0, 1]
        x = np.clip(x, 0.0, 1.0)
        # Clamp y to [0, H]
        y = np.clip(y, 0.0, H)
        # Check left edge: y <= sqrt(3)*x
        max_y_left = np.sqrt(3) * x
        if y > max_y_left:
            y = max_y_left
        # Check right edge: y <= sqrt(3)*(1-x)
        max_y_right = np.sqrt(3) * (1 - x)
        if y > max_y_right:
            y = max_y_right
        # If x is very close to 0 or 1, y must be 0
        if x < 1e-12:
            y = 0.0
        if x > 1 - 1e-12:
            y = 0.0
        pts[i] = [x, y]
    return pts

def _init_symmetric():
    """Create a symmetric initial configuration."""
    pts = np.zeros((N, 2))
    # 3 vertices
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, H]
    # 3 midpoints of edges
    pts[3] = [0.5, 0.0]
    pts[4] = [0.25, H / 2]
    pts[5] = [0.75, H / 2]
    # Centroid
    pts[6] = [0.5, H / 3]
    # 4 more points - place them in a symmetric pattern
    # Two points at 1/4 height on left and right
    h1 = H / 4
    x_left = h1 / np.sqrt(3)
    x_right = 1 - x_left
    pts[7] = [x_left, h1]
    pts[8] = [x_right, h1]
    # Two points at 3/4 height on left and right
    h3 = 3 * H / 4
    x_left3 = h3 / np.sqrt(3)
    x_right3 = 1 - x_left3
    pts[9] = [x_left3, h3]
    pts[10] = [x_right3, h3]
    return pts

def _init_random(rng):
    """Generate random points inside the triangle."""
    pts = np.zeros((N, 2))
    for j in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                pts[j, 0] = v + 0.5 * (1 - u - v)
                pts[j, 1] = H * (1 - u - v)
                break
    return pts

def _refine_all_points(params, deadline):
    """Refine all points one at a time using local search."""
    points = params.reshape(N, 2).copy()
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            # Try small perturbations of point i
            best_x = points[i, 0]
            best_y = points[i, 1]
            best_score = _objective_exact(points.flatten())
            
            # Try Nelder-Mead on just this point
            def obj_i(params_i):
                pts = points.copy()
                pts[i] = params_i
                return _objective_exact(pts.flatten())
            
            try:
                res = minimize(obj_i, [best_x, best_y], method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_score - 1e-12:
                    points[i] = res.x
                    best_score = res.fun
                    improved = True
            except Exception:
                pass
    return points.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    bounds = [(0, 1), (0, H)] * N
    
    best_params = None
    best_score = -np.inf
    
    def update_best(params):
        nonlocal best_params, best_score
        if params is not None:
            pts = _clamp_to_triangle(params.reshape(N, 2))
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params = pts.flatten()
    
    # Phase 1: Global optimization with dual_annealing (smooth objective)
    if time.time() < deadline - 30:
        try:
            init_sym = _init_symmetric().flatten()
            res_da = dual_annealing(
                _objective_smooth,
                bounds,
                x0=init_sym,
                maxiter=100,
                seed=42,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                clip=True
            )
            update_best(res_da.x)
        except Exception:
            pass
    
    # Phase 2: Global optimization with differential_evolution (smooth objective)
    if time.time() < deadline - 20:
        try:
            res_de = differential_evolution(
                _objective_smooth,
                bounds,
                maxiter=50,
                popsize=30,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            update_best(res_de.x)
        except Exception:
            pass
    
    # Phase 3: Local polish with exact objective
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best(res_nm.x)
        except Exception:
            pass
    
    # Phase 4: Coordinate-wise refinement
    if best_params is not None and time.time() < deadline - 5:
        try:
            refined = _refine_all_points(best_params, deadline)
            update_best(refined)
        except Exception:
            pass
    
    # Phase 5: Multi-start local polish with exact objective
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        n_restarts = 3
        for i in range(n_restarts):
            if time.time() > deadline - 5:
                break
            init_pts = _init_random(rng)
            init_params = init_pts.flatten()
            try:
                res_r = minimize(
                    _objective_exact,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                update_best(res_r.x)
            except Exception:
                pass
    
    # Final fallback
    if best_params is None:
        rng = np.random.default_rng(0)
        pts = _init_random(rng)
        best_params = pts.flatten()
    
    points = _clamp_to_triangle(best_params.reshape(N, 2))
    return points
