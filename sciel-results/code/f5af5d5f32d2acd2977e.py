"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
TRI_AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLES.shape[0]

def _min_area(points):
    """Compute the smallest triangle area among all triples."""
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _min_area_grad(points):
    """Compute min area and subgradient."""
    pi = points[_TRIPLES[:, 0]]
    pj = points[_TRIPLES[:, 1]]
    pk = points[_TRIPLES[:, 2]]
    d1 = pj - pi
    d2 = pk - pi
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    i, j, k = _TRIPLES[min_idx]
    pi_pt = points[i]
    pj_pt = points[j]
    pk_pt = points[k]
    
    d1 = pj_pt - pi_pt
    d2 = pk_pt - pi_pt
    c = d1[0] * d2[1] - d1[1] * d2[0]
    sgn = 1.0 if c >= 0 else -1.0
    
    grad = np.zeros((N, 2))
    grad[i] = 0.5 * sgn * np.array([-d2[1], d2[0]])
    grad[j] = 0.5 * sgn * np.array([d2[1], -d2[0]])
    grad[k] = 0.5 * sgn * np.array([-d1[1], d1[0]])
    
    return min_area, grad

def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian."""
    x = bary[:, 0] * 1.0 + bary[:, 1] * 0.5
    y = bary[:, 1] * (np.sqrt(3) / 2.0)
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian to barycentric coords."""
    y = points[:, 1] / (np.sqrt(3) / 2.0)
    x = points[:, 0]
    v = y
    u = x - 0.5 * y
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region."""
    bary = bary.copy()
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _objective_bary(x):
    """Objective: negative min triangle area. x is flat array of 2N barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_area(points)

def _make_constraints():
    """Create linear inequality constraints for barycentric coords."""
    constraints = []
    for i in range(N):
        u_idx = 2 * i
        v_idx = 2 * i + 1
        A_u = np.zeros(2 * N)
        A_u[u_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_u: A @ x})
        A_v = np.zeros(2 * N)
        A_v[v_idx] = 1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_v: A @ x})
        A_sum = np.zeros(2 * N)
        A_sum[u_idx] = -1.0
        A_sum[v_idx] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, A=A_sum: 1.0 + A @ x})
    return constraints

def _fps_init(rng, n_points=N):
    """Farthest Point Sampling initialization."""
    points = []
    # Start with a vertex
    points.append(np.array([0.0, 0.0]))
    
    for _ in range(n_points - 1):
        if len(points) == 0:
            points.append(rng.uniform(0, 1, 2))
            continue
        
        # Find point farthest from all existing points
        best_dist = -1
        best_point = None
        
        # Sample candidate points
        candidates = []
        for _ in range(500):
            p = rng.uniform(0, 1, 2)
            # Check if inside triangle
            if p[1] >= 0 and p[1] <= H and p[0] >= p[1] / np.sqrt(3) and p[0] <= 1 - p[1] / np.sqrt(3):
                candidates.append(p)
        
        candidates = np.array(candidates)
        if len(candidates) == 0:
            candidates = np.array([[0.5, 0.5]])
        
        existing = np.array(points)
        
        # For each candidate, compute min distance to existing points
        for c in candidates:
            dists = np.linalg.norm(existing - c, axis=1)
            min_d = np.min(dists)
            if min_d > best_dist:
                best_dist = min_d
                best_point = c
        
        if best_point is not None:
            points.append(best_point)
    
    return np.array(points[:n_points])

def _structured_init():
    """A well-structured initial configuration based on symmetry."""
    # Use a hexagonal-like pattern
    bary_list = [
        [0.0, 0.0],      # vertex 1
        [1.0, 0.0],      # vertex 2
        [0.0, 1.0],      # vertex 3
        [0.5, 0.0],      # midpoint of edge 1-2
        [0.0, 0.5],      # midpoint of edge 1-3
        [0.5, 0.5],      # midpoint of edge 2-3
        [1/3, 1/3],      # centroid
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.75, 0.25],
    ]
    return np.array(bary_list[:N])

def _local_optimize_slsqp(bary0, constraints, bounds, max_iter=300, ftol=1e-14):
    """Optimize using SLSQP."""
    x0 = bary0.flatten()
    try:
        result = minimize(
            _objective_bary,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        bary_opt = _clamp_bary(result.x.reshape(-1, 2))
        points_opt = _bary_to_cart(bary_opt)
        points_opt = _clip_to_triangle(points_opt)
        return bary_opt, points_opt
    except Exception:
        return bary0, _bary_to_cart(_clamp_bary(bary0))

def _local_optimize_lbfgs(bary0, bounds, max_iter=200):
    """Optimize using L-BFGS-B (no constraints, just bounds)."""
    x0 = bary0.flatten()
    try:
        result = minimize(
            _objective_bary,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14}
        )
        bary_opt = _clamp_bary(result.x.reshape(-1, 2))
        points_opt = _bary_to_cart(bary_opt)
        points_opt = _clip_to_triangle(points_opt)
        return bary_opt, points_opt
    except Exception:
        return bary0, _bary_to_cart(_clamp_bary(bary0))

def _local_optimize_nm(points0, maxiter=500):
    """Optimize using Nelder-Mead in Cartesian coordinates."""
    def obj_cart(x):
        points = x.reshape(-1, 2)
        points = _clip_to_triangle(points)
        return -_min_area(points)
    
    x0 = points0.flatten()
    try:
        result = minimize(
            obj_cart,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        points_opt = _clip_to_triangle(result.x.reshape(-1, 2))
        return points_opt
    except Exception:
        return points0

def _subgradient_refine(points, max_iter=100, base_step=0.02):
    """Subgradient method refinement."""
    points = points.copy()
    current_min = _min_area(points)
    
    for it in range(max_iter):
        min_area, grad = _min_area_grad(points)
        
        # Adaptive step size
        step = base_step / (1 + 0.1 * it)
        new_points = points + step * grad
        new_points = _clip_to_triangle(new_points)
        
        new_min = _min_area(new_points)
        if new_min > current_min:
            points = new_points
            current_min = new_min
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-10:
                break
    
    return points, current_min

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    constraints = _make_constraints()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_score = -np.inf
    best_points = None
    best_bary = None
    
    # Generate diverse initializations
    inits_bary = []
    inits_points = []
    
    # Structured initialization
    bary_struct = _structured_init()
    inits_bary.append(bary_struct)
    inits_points.append(_bary_to_cart(bary_struct))
    
    # FPS initializations with different seeds
    rng = np.random.RandomState(42)
    for seed in range(10):
        if time_left() < 10.0:
            break
        rng_fps = np.random.RandomState(seed * 12345 + 67890)
        pts = _fps_init(rng_fps)
        inits_points.append(pts)
        bary = _cart_to_bary(pts)
        bary = _clamp_bary(bary)
        inits_bary.append(bary)
    
    # Perturbed structured inits
    rng_pert = np.random.RandomState(999)
    for scale in [0.05, 0.1, 0.15, 0.25]:
        if time_left() < 10.0:
            break
        bary_pert = bary_struct + rng_pert.normal(0, scale, size=bary_struct.shape)
        bary_pert = _clamp_bary(bary_pert)
        inits_bary.append(bary_pert)
        inits_points.append(_bary_to_cart(bary_pert))
    
    # Optimize each initialization
    for i, (bary0, pts0) in enumerate(zip(inits_bary, inits_points)):
        if time_left() < 20.0:
            break
        
        # SLSQP optimization
        bary_opt, points_opt = _local_optimize_slsqp(bary0, constraints, bounds, max_iter=200)
        score = _min_area(points_opt)
        if score > best_score:
            best_score = score
            best_points = points_opt.copy()
            best_bary = bary_opt.copy()
        
        # L-BFGS-B optimization
        if time_left() > 15.0:
            bary_lbfgs, points_lbfgs = _local_optimize_lbfgs(bary0, bounds, max_iter=150)
            score_lbfgs = _min_area(points_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_points = points_lbfgs.copy()
                best_bary = bary_lbfgs.copy()
        
        # Nelder-Mead in Cartesian
        if time_left() > 15.0:
            points_nm = _local_optimize_nm(pts0, maxiter=300)
            score_nm = _min_area(points_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = points_nm.copy()
                best_bary = _cart_to_bary(points_nm)
                best_bary = _clamp_bary(best_bary)
    
    # Subgradient refinement from best
    if time_left() > 20.0 and best_points is not None:
        for base_step in [0.03, 0.01, 0.005]:
            if time_left() < 10.0:
                break
            pts_ref, score_ref = _subgradient_refine(best_points, max_iter=50, base_step=base_step)
            if score_ref > best_score:
                best_score = score_ref
                best_points = pts_ref.copy()
                best_bary = _cart_to_bary(best_points)
                best_bary = _clamp_bary(best_bary)
    
    # Final SLSQP polish
    if time_left() > 15.0 and best_bary is not None:
        bary_final, points_final = _local_optimize_slsqp(best_bary, constraints, bounds, max_iter=150, ftol=1e-15)
        score_final = _min_area(points_final)
        if score_final > best_score:
            best_score = score_final
            best_points = points_final.copy()
            best_bary = bary_final.copy()
    
    # One more round of diversification and optimization if time allows
    if time_left() > 30.0 and best_points is not None:
        rng_div = np.random.RandomState(777)
        for scale in [0.02, 0.05, 0.1]:
            if time_left() < 15.0:
                break
            bary_div = best_bary + rng_div.normal(0, scale, size=best_bary.shape)
            bary_div = _clamp_bary(bary_div)
            bary_opt2, points_opt2 = _local_optimize_slsqp(bary_div, constraints, bounds, max_iter=150)
            score2 = _min_area(points_opt2)
            if score2 > best_score:
                best_score = score2
                best_points = points_opt2.copy()
                best_bary = bary_opt2.copy()
    
    if best_points is None:
        best_points = _bary_to_cart(_structured_init())
        best_points = _clip_to_triangle(best_points)
    
    return best_points
