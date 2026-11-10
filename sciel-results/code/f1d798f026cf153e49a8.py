# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

# Precompute triangle index arrays (286 triangles for 13 points)
_N = 13
_TRIS = np.array([(i, j, k) for i in range(_N) for j in range(i+1, _N) for k in range(j+1, _N)], dtype=np.int32)
_I_IDX = _TRIS[:, 0]
_J_IDX = _TRIS[:, 1]
_K_IDX = _TRIS[:, 2]


def _triangle_areas(pts):
    """Compute all 286 triangle areas vectorized using precomputed indices."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas


def _score(pts):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    if len(pts) < 3:
        return 0.0
    areas = _triangle_areas(pts)
    a_min = np.min(areas)
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _soft_min_and_grad(pts, beta):
    """
    Compute soft-min of triangle areas and its gradient w.r.t. point coordinates.
    Returns (soft_min_value, grad_26).
    """
    areas = _triangle_areas(pts)
    areas = np.maximum(areas, 1e-15)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_shifted = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_val + np.log(sum_exp)
    soft_min = -log_sum / beta
    
    # Weights: w_i = exp(-beta * a_i) / sum(exp(-beta * a_j))
    weights = exp_shifted / sum_exp  # shape (286,)
    
    # Gradient of soft_min w.r.t. each coordinate
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    grad = np.zeros(2 * _N)
    
    dx_i = signs * (y2 - y3) / 2.0
    dy_i = signs * (x3 - x2) / 2.0
    dx_j = signs * (y3 - y1) / 2.0
    dy_j = signs * (x3 - x1) / 2.0
    dx_k = signs * (y1 - y2) / 2.0
    dy_k = signs * (x2 - x1) / 2.0
    
    w = weights  # (286,)
    
    np.add.at(grad, _I_IDX, w * dx_i)
    np.add.at(grad, _I_IDX + _N, w * dy_i)
    np.add.at(grad, _J_IDX, w * dx_j)
    np.add.at(grad, _J_IDX + _N, w * dy_j)
    np.add.at(grad, _K_IDX, w * dx_k)
    np.add.at(grad, _K_IDX + _N, w * dy_k)
    
    return soft_min, grad


def _hull_area_and_grad(pts):
    """Compute convex hull area and its gradient w.r.t. point coordinates."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return None, None
    
    # Gradient of hull area: only hull vertices contribute
    # For a convex polygon, d(area)/d(x_i) = 0.5 * (y_{i+1} - y_{i-1}) for hull vertex i
    # where neighbors are in hull order
    hull_vertices = hull.vertices  # ordered around the hull
    n_hull = len(hull_vertices)
    
    if n_hull < 3:
        return hull_area, None
    
    grad = np.zeros(2 * _N)
    
    for idx, v in enumerate(hull_vertices):
        prev_v = hull_vertices[(idx - 1) % n_hull]
        next_v = hull_vertices[(idx + 1) % n_hull]
        
        # d(area)/d(x_v) = 0.5 * (y_next - y_prev)
        # d(area)/d(y_v) = 0.5 * (x_prev - x_next)
        grad[v] = 0.5 * (pts[next_v, 1] - pts[prev_v, 1])
        grad[v + _N] = 0.5 * (pts[prev_v, 0] - pts[next_v, 0])
    
    return hull_area, grad


def _objective_and_grad(x, beta):
    """
    Objective: -soft_min / hull_area, and its full gradient.
    d(-s/h)/dx = -(grad_s * h - s * grad_h) / h^2 = -grad_s/h + s*grad_h/h^2
    """
    pts = np.column_stack([x[0:_N], x[_N:2*_N]])
    soft_min, grad_soft = _soft_min_and_grad(pts, beta)
    hull_area, grad_hull = _hull_area_and_grad(pts)
    
    if hull_area is None or hull_area <= 0 or not np.isfinite(soft_min):
        return 1e10, np.zeros(2*_N)
    
    obj = -soft_min / hull_area
    
    # Full gradient: d(-s/h)/dx = -grad_s/h + s*grad_h/h^2
    if grad_hull is not None:
        grad = -grad_soft / hull_area + soft_min * grad_hull / (hull_area * hull_area)
    else:
        grad = -grad_soft / hull_area
    
    return obj, grad


def _regular_polygon(n=13):
    """Create regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _fast_relax(pts, n_iters=50, learning_rate=0.01, start_time=None, time_limit=600):
    """
    Fast vectorized relaxation: move internal points to maximize min triangle area.
    Uses force-based approach with vectorized operations.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    n = len(pts)
    
    # Identify hull vertices
    try:
        hull = ConvexHull(pts)
        hull_mask = np.zeros(n, dtype=bool)
        hull_mask[hull.vertices] = True
    except Exception:
        return pts
    
    internal_mask = ~hull_mask
    internal_indices = np.where(internal_mask)[0]
    
    if len(internal_indices) == 0:
        return pts
    
    for iteration in range(n_iters):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Compute all triangle areas
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        threshold = min(areas.max() * 0.3, min_area * 3)
        
        # Find triangles with area below threshold
        small_mask = areas < threshold
        if not np.any(small_mask):
            break
        
        # Compute forces for internal vertices
        forces = np.zeros_like(pts)
        
        # For each small triangle, push internal vertices away from centroid
        small_tris = _TRIS[small_mask]
        small_areas = areas[small_mask]
        
        for idx in range(len(small_tris)):
            i, j, k = small_tris[idx]
            tri_area = small_areas[idx]
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            strength = threshold - tri_area
            
            for v in [i, j, k]:
                if internal_mask[v]:
                    direction = pts[v] - centroid
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                        forces[v] += learning_rate * strength * direction
        
        # Apply forces only to internal vertices
        pts[internal_mask] += 0.5 * forces[internal_mask]
    
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    time_limit = 600  # Safety margin from 630s

    n = 13
    best_pts = _regular_polygon(n)
    best_score = _score(best_pts)

    # Try multiple initializations with different structures
    initializations = []

    # 1. Regular polygon
    initializations.append(_regular_polygon(n))

    # 2. Two concentric rings
    for inner_n, outer_n in [(5, 8), (6, 7), (4, 9)]:
        inner_angles = np.linspace(0, 2*np.pi, inner_n, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, outer_n, endpoint=False) + np.pi/outer_n
        inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initializations.append(np.vstack([inner_pts, outer_pts]))

    # 3. Grid-like arrangement
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([0.3 + 0.4*i, 0.3 + 0.4*j])
    grid_pts = np.array(grid_pts[:n])
    initializations.append(grid_pts)

    # 4. Slightly perturbed regular polygon
    for seed in range(5):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.1, (n, 2))
        initializations.append(perturbed)

    # 5. Another set of perturbations
    for seed in range(10, 15):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.15, (n, 2))
        initializations.append(perturbed)

    # 6. Random initializations with quick relaxation
    for seed in range(20, 25):
        rng2 = np.random.default_rng(seed)
        rand_pts = rng2.random((n, 2))
        rand_pts = _fast_relax(rand_pts, n_iters=30, learning_rate=0.02, start_time=start_time, time_limit=time_limit)
        initializations.append(rand_pts)

    # Optimize each initialization
    beta_schedule = [20, 50, 150, 400, 1000, 3000, 10000]

    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 40:
            break

        # Flatten to 26-d vector
        x0 = init_pts.flatten()

        # Anneal beta
        current_x = x0.copy()
        for beta in beta_schedule:
            elapsed = time.time() - start_time
            if elapsed > time_limit - 20:
                break

            # L-BFGS-B with analytical gradient
            try:
                result = minimize(
                    lambda x, b=beta: _objective_and_grad(x, b)[0],
                    current_x,
                    method='L-BFGS-B',
                    jac=lambda x, b=beta: _objective_and_grad(x, b)[1],
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
                if result.fun < _objective_and_grad(current_x, beta)[0] - 1e-12:
                    current_x = result.x
            except Exception:
                pass

            # Quick Nelder-Mead polish
            try:
                nm_result = minimize(
                    lambda x, b=beta: _objective_and_grad(x, b)[0],
                    current_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if nm_result.fun < _objective_and_grad(current_x, beta)[0] - 1e-12:
                    current_x = nm_result.x
            except Exception:
                pass

            # Fast relaxation after gradient step
            current_pts = np.column_stack([current_x[0:_N], current_x[_N:2*_N]])
            relaxed_pts = _fast_relax(current_pts, n_iters=30, learning_rate=0.005, start_time=start_time, time_limit=time_limit)
            relaxed_score = _score(relaxed_pts)
            if relaxed_score > _score(current_pts):
                current_x = relaxed_pts.flatten()

        # Evaluate final result
        final_pts = np.column_stack([current_x[0:_N], current_x[_N:2*_N]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10 and best_score > 0:
        x_final = best_pts.flatten()

        # Use a high-beta soft-min for near-exact behavior with smooth gradients
        beta_final = 20000

        def final_obj(x):
            return _objective_and_grad(x, beta_final)[0]

        def final_jac(x):
            return _objective_and_grad(x, beta_final)[1]

        # L-BFGS-B with high beta
        try:
            result = minimize(
                final_obj,
                x_final,
                method='L-BFGS-B',
                jac=final_jac,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-8,
                    'maxls': 30
                }
            )
            if result.fun < final_obj(x_final) - 1e-14:
                x_final = result.x
        except Exception:
            pass

        # Nelder-Mead with high beta
        try:
            nm_final = minimize(
                final_obj,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            if nm_final.fun < final_obj(x_final) - 1e-14:
                x_final = nm_final.x
        except Exception:
            pass

        # Final fast relaxation
        final_pts_tmp = np.column_stack([x_final[0:_N], x_final[_N:2*_N]])
        relaxed_final = _fast_relax(final_pts_tmp, n_iters=50, learning_rate=0.002, start_time=start_time, time_limit=time_limit)
        if _score(relaxed_final) > _score(final_pts_tmp):
            x_final = relaxed_final.flatten()

        final_pts = np.column_stack([x_final[0:_N], x_final[_N:2*_N]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Validate result
    if best_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _regular_polygon(n)

    return best_pts


# EVOLVE-BLOCK-END
