# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

_N = 13
_TRIS = np.array([(i, j, k) for i in range(_N) for j in range(i+1, _N) for k in range(j+1, _N)], dtype=np.int32)
_I_IDX = _TRIS[:, 0]
_J_IDX = _TRIS[:, 1]
_K_IDX = _TRIS[:, 2]


def _triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
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
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_shifted = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_val + np.log(sum_exp)
    soft_min = -log_sum / beta
    
    weights = exp_shifted / sum_exp
    
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
    
    w = weights
    
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
    
    hull_vertices = hull.vertices
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
    Objective: -soft_min / hull_area, and its gradient.
    """
    pts = np.column_stack([x[0:_N], x[_N:2*_N]])
    soft_min, grad_soft = _soft_min_and_grad(pts, beta)
    
    hull_area, grad_hull = _hull_area_and_grad(pts)
    
    if hull_area is None or hull_area <= 0 or not np.isfinite(soft_min) or not np.isfinite(hull_area):
        return 1e10, np.zeros(2*_N)
    
    obj = -soft_min / hull_area
    
    if grad_hull is not None:
        # d(-sm/ha)/dx = -dsm/dx * (1/ha) + sm * dha/dx / ha^2
        grad = -grad_soft / hull_area + soft_min * grad_hull / (hull_area * hull_area)
    else:
        grad = -grad_soft / hull_area
    
    return obj, grad


def _regular_polygon(n=13):
    """Create regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _hexagonal_lattice_init():
    """Create a hexagonal lattice arrangement for 13 points."""
    # Center + 6 around + 6 more
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (6 points, offset)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        pts.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
    return np.array(pts)


def _fast_relax(pts, n_iters=200, lr=0.01, start_time=None, time_limit=600):
    """
    Fast iterative relaxation: move each point to maximize its local contribution
    to the minimum triangle area, using coordinate-wise optimization.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    n = len(pts)
    
    # Identify hull vertices
    try:
        hull = ConvexHull(pts)
        hull_vertices = set(hull.vertices)
    except Exception:
        return pts
    
    internal_indices = [i for i in range(n) if i not in hull_vertices]
    
    for iteration in range(n_iters):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Compute current areas
        areas = _triangle_areas(pts)
        min_area = np.min(areas)
        
        # For each internal point, try to move it to improve the worst triangles it's in
        for v in internal_indices:
            # Find triangles containing point v
            tri_indices = np.where((_I_IDX == v) | (_J_IDX == v) | (_K_IDX == v))[0]
            if len(tri_indices) == 0:
                continue
            
            # Get the worst triangle involving v
            tri_areas = areas[tri_indices]
            worst_local = np.min(tri_areas)
            
            # Try small perturbations
            best_pos = pts[v].copy()
            best_score_local = worst_local
            
            for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                    new_pos = pts[v] + np.array([dx, dy])
                    pts_temp = pts.copy()
                    pts_temp[v] = new_pos
                    
                    # Check if this improves the local minimum
                    tri_areas_temp = _triangle_areas(pts_temp)[tri_indices]
                    local_min = np.min(tri_areas_temp)
                    
                    if local_min > best_score_local + 1e-10:
                        best_score_local = local_min
                        best_pos = new_pos
            
            pts[v] = best_pos
        
        # Apply a small amount of Laplacian smoothing to internal points
        try:
            tri = Delaunay(pts)
            for v in internal_indices:
                neighbors = []
                for simplex in tri.simplices:
                    if v in simplex:
                        for other in simplex:
                            if other != v:
                                neighbors.append(other)
                if neighbors:
                    neighbor_avg = np.mean(pts[neighbors], axis=0)
                    pts[v] = (1 - lr) * pts[v] + lr * neighbor_avg
        except Exception:
            pass
    
    return pts


def _simulated_annealing(pts, start_temp=0.1, end_temp=0.001, n_iters=500, start_time=None, time_limit=600):
    """
    Simulated annealing to escape local minima.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    n = len(pts)
    
    for iteration in range(n_iters):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Temperature schedule
        t = iteration / n_iters
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Random perturbation
        rng = np.random.default_rng(iteration + 1000)
        new_pts = pts + rng.normal(0, temp, (n, 2))
        
        # Ensure no points are too close to each other
        new_score = _score(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            # Accept with probability exp((new - current) / temp)
            delta = new_score - current_score
            if delta < 0 and temp > 0:
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    pts = new_pts
                    current_score = new_score
    
    return best_pts


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

    # 2. Hexagonal lattice
    initializations.append(_hexagonal_lattice_init())

    # 3. Two concentric rings
    for inner_n, outer_n in [(5, 8), (6, 7), (4, 9)]:
        inner_angles = np.linspace(0, 2*np.pi, inner_n, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, outer_n, endpoint=False) + np.pi/outer_n
        inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initializations.append(np.vstack([inner_pts, outer_pts]))

    # 4. Slightly perturbed regular polygon
    for seed in range(5):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.1, (n, 2))
        initializations.append(perturbed)

    # 5. More perturbations
    for seed in range(10, 15):
        rng2 = np.random.default_rng(seed)
        perturbed = _regular_polygon(n)
        perturbed += rng2.normal(0, 0.15, (n, 2))
        initializations.append(perturbed)

    # 6. Random starts
    for seed in range(20, 25):
        rng2 = np.random.default_rng(seed)
        rand_pts = rng2.random((n, 2))
        initializations.append(rand_pts)

    # Optimize each initialization
    beta_schedule = [30, 80, 200, 500, 1500, 5000]

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
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-6,
                        'maxls': 30
                    }
                )
                if result.fun < _objective_and_grad(current_x, beta)[0] - 1e-12:
                    current_x = result.x
            except Exception:
                pass

            # Nelder-Mead polish
            try:
                nm_result = minimize(
                    lambda x, b=beta: _objective_and_grad(x, b)[0],
                    current_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1500,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if nm_result.fun < _objective_and_grad(current_x, beta)[0] - 1e-12:
                    current_x = nm_result.x
            except Exception:
                pass

            # Fast relaxation
            current_pts = np.column_stack([current_x[0:_N], current_x[_N:2*_N]])
            relaxed_pts = _fast_relax(current_pts, n_iters=50, lr=0.005, start_time=start_time, time_limit=time_limit)
            relaxed_score = _score(relaxed_pts)
            if relaxed_score > _score(current_pts):
                current_x = relaxed_pts.flatten()

        # Evaluate final result
        final_pts = np.column_stack([current_x[0:_N], current_x[_N:2*_N]])
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

        # Try simulated annealing from this point
        elapsed = time.time() - start_time
        if elapsed < time_limit - 30 and final_score > 0:
            sa_pts = _simulated_annealing(final_pts, start_temp=0.05, end_temp=0.001, n_iters=200, start_time=start_time, time_limit=time_limit)
            sa_score = _score(sa_pts)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts

    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10 and best_score > 0:
        x_final = best_pts.flatten()

        # Use a high-beta soft-min for near-exact behavior with smooth gradients
        beta_final = 10000

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
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-8,
                    'maxls': 50
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
                    'maxiter': 5000,
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
        relaxed_final = _fast_relax(final_pts_tmp, n_iters=100, lr=0.002, start_time=start_time, time_limit=time_limit)
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
