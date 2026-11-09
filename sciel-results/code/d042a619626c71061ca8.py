"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

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

def _all_areas(points):
    """Compute all triangle areas. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples."""
    return np.min(_all_areas(points))

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    for i in range(N):
        x, y = pts[i]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, H)
        max_y_left = SQRT3 * x
        if y > max_y_left:
            y = max_y_left
        max_y_right = SQRT3 * (1 - x)
        if y > max_y_right:
            y = max_y_right
        if x < 1e-12:
            y = 0.0
        if x > 1 - 1e-12:
            y = 0.0
        pts[i] = [x, y]
    return pts

def _params_to_points_symmetric(params):
    """
    Map symmetric parameters to 11 points with reflection symmetry about x=0.5.
    Structure:
    - Point 0: on the symmetry axis (x=0.5), parameterized by y only
    - Points 1-10: 5 pairs of mirrored points.
    params: array of shape (11,)
      params[0] = y of the axis point
      params[1+2i], params[2+2i] = x, y for left point of pair i
    """
    pts = np.zeros((N, 2))
    pts[0] = [0.5, params[0]]
    for i in range(5):
        x = params[1 + 2*i]
        y = params[2 + 2*i]
        pts[1 + 2*i] = [x, y]
        pts[2 + 2*i] = [1.0 - x, y]
    return pts

def _points_to_params_symmetric(points):
    """Extract symmetric parameters from a full point set."""
    pts = points.copy()
    params = np.zeros(11)
    params[0] = pts[0, 1]
    for i in range(5):
        left_idx = 1 + 2*i
        right_idx = 2 + 2*i
        x_left = min(pts[left_idx, 0], pts[right_idx, 0])
        y_avg = (pts[left_idx, 1] + pts[right_idx, 1]) / 2.0
        params[1 + 2*i] = x_left
        params[2 + 2*i] = y_avg
    return params

def _inside_penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 * (1 - x) - y
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def _objective_symmetric(params):
    """Objective in symmetric parameter space: -min_area + penalty."""
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 0.1:
        return 1e6 + pen
    area = _min_area(pts)
    return -area + 5.0 * pen

def _smooth_min_area_symmetric(params, beta=50.0):
    """Smooth min area in symmetric parameter space."""
    pts = _params_to_points_symmetric(params)
    areas = _all_areas(pts)
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    logsumexp_val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -logsumexp_val / beta

def _objective_smooth_symmetric(params, beta=50.0):
    """Smooth objective in symmetric parameter space."""
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 0.1:
        return 1e6 + pen
    smin = _smooth_min_area_symmetric(params, beta)
    return -smin + 5.0 * pen

def _init_symmetric_params():
    """Create initial symmetric parameters."""
    params = np.zeros(11)
    params[0] = H / 3.0
    heights = [H * 0.15, H * 0.3, H * 0.5, H * 0.7, H * 0.85]
    for i, h in enumerate(heights):
        x_min = h / SQRT3
        x = (x_min + 0.5) / 2.0
        params[1 + 2*i] = x
        params[2 + 2*i] = h
    return params

def _init_random_symmetric(rng):
    """Generate random symmetric parameters."""
    params = np.zeros(11)
    params[0] = rng.random() * H
    for i in range(5):
        y = rng.random() * H
        x_min = y / SQRT3
        x_max = 0.5
        if x_min >= x_max:
            x = 0.5
        else:
            x = x_min + rng.random() * (x_max - x_min)
        params[1 + 2*i] = x
        params[2 + 2*i] = y
    return params

def _refine_symmetric(params, deadline):
    """Refine symmetric parameters using Nelder-Mead on individual coordinates."""
    p = params.copy()
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(11):
            if time.time() > deadline - 5:
                break
            best_val = _objective_symmetric(p)
            
            def obj_i(param_i):
                p[i] = param_i
                return _objective_symmetric(p)
            
            try:
                res = minimize(obj_i, [p[i]], method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val - 1e-12:
                    p[i] = res.x[0]
                    improved = True
            except Exception:
                pass
    return p

def _simulated_annealing_symmetric(params0, deadline, rng, initial_temp=10.0, cooling=0.999, max_iter=50000):
    """
    Simulated Annealing in symmetric parameter space.
    Uses a smooth objective for the acceptance criterion to handle non-differentiability.
    """
    current_params = params0.copy()
    current_score = -_objective_symmetric(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    bounds_sym = [(0, H)]
    for i in range(5):
        bounds_sym.append((0, 0.5))
        bounds_sym.append((0, H))
    
    temp = initial_temp
    iters = 0
    
    while time.time() < deadline - 5 and iters < max_iter:
        iters += 1
        # Generate neighbor
        new_params = current_params + rng.normal(0, 0.05, size=11)
        # Clip to bounds
        for i, b in enumerate(bounds_sym):
            new_params[i] = np.clip(new_params[i], b[0], b[1])
        
        new_score = -_objective_symmetric(new_params)
        
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                accept_prob = np.exp(delta / temp)
                if rng.random() < accept_prob:
                    current_params = new_params
                    current_score = new_score
        
        temp *= cooling
        if temp < 1e-6:
            break
            
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Bounds for symmetric parameters
    bounds_sym = [(0, H)]
    for i in range(5):
        bounds_sym.append((0, 0.5))
        bounds_sym.append((0, H))
    
    best_params_sym = None
    best_score = -np.inf
    
    def update_best_sym(params_sym):
        nonlocal best_params_sym, best_score
        if params_sym is not None:
            pts = _clamp_to_triangle(_params_to_points_symmetric(params_sym))
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = params_sym.copy()
    
    def update_best_full(points):
        nonlocal best_params_sym, best_score
        if points is not None:
            pts = _clamp_to_triangle(points)
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_params_sym = _points_to_params_symmetric(pts)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global optimization in symmetric space with dual_annealing
    if time.time() < deadline - 30:
        try:
            init_sym = _init_symmetric_params()
            res_da = dual_annealing(
                _objective_smooth_symmetric,
                bounds_sym,
                x0=init_sym,
                maxiter=150,
                seed=42,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                clip=True
            )
            update_best_sym(res_da.x)
        except Exception:
            pass
    
    # Phase 2: Global optimization with differential_evolution in symmetric space
    if time.time() < deadline - 25:
        try:
            res_de = differential_evolution(
                _objective_smooth_symmetric,
                bounds_sym,
                maxiter=80,
                popsize=40,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            update_best_sym(res_de.x)
        except Exception:
            pass
    
    # Phase 3: Simulated Annealing refinement (strong escape from local minima)
    if best_params_sym is not None and time.time() < deadline - 10:
        try:
            sa_params, sa_score = _simulated_annealing_symmetric(
                best_params_sym, deadline, rng, 
                initial_temp=5.0, cooling=0.9995, max_iter=20000
            )
            update_best_sym(sa_params)
        except Exception:
            pass
    
    # Phase 4: Local polish with exact objective in symmetric space
    if best_params_sym is not None and time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective_symmetric,
                best_params_sym,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best_sym(res_nm.x)
        except Exception:
            pass
    
    # Phase 5: Coordinate-wise refinement in symmetric space
    if best_params_sym is not None and time.time() < deadline - 5:
        try:
            refined = _refine_symmetric(best_params_sym, deadline)
            update_best_sym(refined)
        except Exception:
            pass
    
    # Phase 6: Multi-start in symmetric space
    if time.time() < deadline - 15:
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 5:
                break
            init_params = _init_random_symmetric(rng)
            try:
                res_r = minimize(
                    _objective_symmetric,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                update_best_sym(res_r.x)
            except Exception:
                pass
    
    # Final: also try full-space local polish from best symmetric result
    if best_params_sym is not None and time.time() < deadline - 5:
        try:
            pts_full = _params_to_points_symmetric(best_params_sym)
            pts_clamped = _clamp_to_triangle(pts_full)
            
            def obj_full(params_full):
                pts = _clamp_to_triangle(params_full.reshape(N, 2))
                pen = _inside_penalty(pts)
                if pen > 0.1:
                    return 1e6 + pen
                return -_min_area(pts) + 5.0 * pen
            
            res_full = minimize(
                obj_full,
                pts_clamped.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            update_best_full(res_full.x.reshape(N, 2))
        except Exception:
            pass
    
    # Final fallback
    if best_params_sym is None:
        best_params_sym = _init_symmetric_params()
    
    points = _clamp_to_triangle(_params_to_points_symmetric(best_params_sym))
    return points
