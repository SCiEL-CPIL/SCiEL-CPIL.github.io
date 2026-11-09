"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

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

def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _all_areas(points):
    """Compute all 165 triangle areas."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * cross

def _params_to_points(params):
    """
    Convert 2N parameters to N points inside the triangle.
    Use the mapping: for each point, generate (u, v) in [0,1]^2 with u+v<=1,
    then map to triangle coordinates.
    To make this smooth and differentiable, we use:
      x = 0.5*(1-u-v) + v  ... wait, let's use a cleaner mapping.
    
    Standard mapping from (u,v) in {u>=0, v>=0, u+v<=1} to triangle:
      point = u*A + v*B + (1-u-v)*C
    But we need u,v in [0,1] with u+v<=1.
    
    Alternative: use two parameters per point in [0,1], and map via:
      Let s = u, t = v*(1-u). Then s in [0,1], t in [0, 1-s].
      Barycentric: (s, t, 1-s-t)
      point = s*A + t*B + (1-s-t)*C
    
    This ensures all points are inside the triangle.
    """
    u = params[0::2]  # N values in [0,1]
    v = params[1::2]  # N values in [0,1]
    # Map to barycentric coordinates
    s = u
    t = v * (1.0 - u)
    # Now s >= 0, t >= 0, s + t <= 1 (since t <= 1-u, so s+t <= u + 1-u = 1)
    # Barycentric: (s, t, 1-s-t) relative to A, B, C
    # point = s*A + t*B + (1-s-t)*C
    x = s * 0.0 + t * 1.0 + (1.0 - s - t) * 0.5
    y = s * 0.0 + t * 0.0 + (1.0 - s - t) * H
    points = np.column_stack([x, y])
    return points

def _objective(params):
    """Objective: minimize -min_area (i.e., maximize min area)."""
    points = _params_to_points(params)
    area = _min_area(points)
    return -area

def _is_inside(points, eps=1e-10):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = y + eps
    d2 = np.sqrt(3) * x - y + eps
    d3 = np.sqrt(3) * (1.0 - x) - y + eps
    return np.all(d1 >= 0) and np.all(d2 >= 0) and np.all(d3 >= 0)

def _project_point(x, y):
    """Project a point onto the triangle (clamp to nearest boundary if outside)."""
    # The triangle is defined by:
    # y >= 0
    # y <= sqrt(3)*x
    # y <= sqrt(3)*(1-x)
    # If outside, clamp to the nearest edge.
    # Simple approach: if any constraint violated, adjust.
    # This is a rough projection for fallback use.
    if y < 0:
        y = 0
    max_y_left = np.sqrt(3) * x
    max_y_right = np.sqrt(3) * (1.0 - x)
    if y > max_y_left:
        y = max_y_left
    if y > max_y_right:
        y = max_y_right
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Bounds: each parameter in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution (global search)
    # Run multiple DE instances with different seeds
    de_configs = [
        {'maxiter': 80, 'popsize': 30, 'seed': 42},
        {'maxiter': 60, 'popsize': 25, 'seed': 123},
        {'maxiter': 50, 'popsize': 20, 'seed': 777},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 30:
            break
        try:
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=1e-8,
                seed=cfg['seed'],
                polish=False,
                init='latinhypercube'
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if best_params is not None:
        if time.time() < deadline - 15:
            try:
                res_nm = minimize(
                    _objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass
        
        # Additional Nelder-Mead with different initial simplex size
        if time.time() < deadline - 10:
            try:
                # Perturb slightly and refine
                rng = np.random.default_rng(999)
                perturbed = best_params + rng.normal(0, 0.01, size=best_params.shape)
                perturbed = np.clip(perturbed, 0, 1)
                res_nm2 = minimize(
                    _objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res_nm2.fun < best_score:
                    best_score = res_nm2.fun
                    best_params = res_nm2.x.copy()
            except Exception:
                pass
    
    # Phase 3: Multi-start local search with random restarts
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for i in range(8):
            if time.time() > deadline - 5:
                break
            # Random parameters in [0,1]
            init_params = rng.random(2 * N)
            try:
                res_r = minimize(
                    _objective,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_r.fun < best_score:
                    best_score = res_r.fun
                    best_params = res_r.x.copy()
            except Exception:
                pass
    
    # Fallback if nothing worked
    if best_params is None:
        # Use a symmetric initial configuration
        best_params = np.zeros(2 * N)
        # Place points in a grid-like pattern
        # Row 0: 1 point at top
        # Row 1: 2 points
        # Row 2: 3 points  
        # Row 3: 5 points
        # Total: 11
        # In barycentric (s, t) where point = s*A + t*B + (1-s-t)*C
        # Top vertex C: s=0, t=0
        # Bottom left A: s=1, t=0
        # Bottom right B: s=0, t=1
        
        positions = [
            (0.5, 0.25),   # near top
            (0.25, 0.5),   # left
            (0.75, 0.5),   # right
            (0.1, 0.1),    # bottom left area
            (0.5, 0.1),    # bottom center
            (0.9, 0.1),    # bottom right area
            (0.3, 0.3),
            (0.7, 0.3),
            (0.5, 0.5),
            (0.2, 0.7),
            (0.8, 0.7),
        ]
        for i, (s, t) in enumerate(positions):
            best_params[2*i] = s
            best_params[2*i+1] = t
    
    # Convert to points
    points = _params_to_points(best_params)
    
    # Final safety: verify all points are inside, project if needed
    for i in range(N):
        x, y = points[i]
        # Check and clamp
        if y < 0:
            y = 0
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x
        if y > np.sqrt(3) * (1.0 - x):
            y = np.sqrt(3) * (1.0 - x)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, H)
        points[i] = [x, y]
    
    return points
