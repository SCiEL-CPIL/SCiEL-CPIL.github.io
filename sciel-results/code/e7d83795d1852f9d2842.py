import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < -eps:
            points[m, 1] = 0.0
        if y + SQRT3 * x > SQRT3 + eps:
            d = (y + SQRT3 * x - SQRT3) / 2.0
            if d > 0:
                points[m, 0] -= SQRT3 * d
                points[m, 1] -= d
        if y > SQRT3 * x + eps:
            d = (SQRT3 * x - y) / 2.0
            if d < 0:
                points[m, 0] += SQRT3 * d
                points[m, 1] -= d
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < 0:
            points[m, 1] = 0
        if y > SQRT3 * x:
            if x > 0:
                points[m, 1] = SQRT3 * x
            else:
                points[m, 1] = 0
        if y > SQRT3 * (1 - x):
            if x < 1:
                points[m, 1] = SQRT3 * (1 - x)
            else:
                points[m, 1] = 0
    return points

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 4
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    bary = np.column_stack([u, v])
    return _bary_to_cart(bary)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _objective(params, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points.copy())
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return points, _min_area(points)
    
    new_points = result.x.reshape(N, 2)
    new_points = _project_to_triangle(new_points)
    score = _min_area(new_points)
    
    if score > _min_area(points):
        return new_points, score
    else:
        return points, _min_area(points)

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 60.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(20):
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _initial_guess_grid():
    """Generate a structured grid-like initial configuration."""
    positions = []
    for i in range(3):
        x = 0.1 + i * 0.3
        positions.append((x, 0.05))
    for i in range(4):
        x = 0.05 + i * 0.25
        y = 0.25
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    for i in range(3):
        x = 0.15 + i * 0.25
        y = 0.45
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    positions.append((0.5, 0.6))
    
    valid_positions = []
    for p in positions:
        x, y = p
        if y >= -1e-10 and y <= SQRT3 * x + 1e-10 and y <= SQRT3 * (1 - x) + 1e-10:
            valid_positions.append(p)
    
    if len(valid_positions) >= N:
        return np.array(valid_positions[:N])
    else:
        return _sobol_points_in_triangle(N, seed=0)

def _lagrangian_optimize(points, time_limit=20.0, max_dual_iters=50, tol=1e-8):
    """
    Lagrangian relaxation with subgradient optimization.
    
    Problem: maximize t subject to area_i >= t for all i, points in triangle.
    Lagrangian: L(x, lambda, t) = t + sum_i lambda_i * (area_i(x) - t)
    where lambda_i >= 0.
    
    Dual: maximize over lambda >= 0, minimize over x (in triangle) of L.
    """
    start_t = time.time()
    current_points = points.copy()
    best_points = points.copy()
    best_score = _min_area(points)
    
    # Initialize Lagrange multipliers
    lam = np.zeros(N_TRI)
    # Learning rate for dual updates
    lr = 0.01
    # Regularization parameter for primal step
    reg = 0.01
    
    for dual_iter in range(max_dual_iters):
        if time.time() - start_t > time_limit:
            break
        
        # Primal step: minimize -L(x, lam, t) over x in triangle
        # L(x, lam, t) = t + sum_i lam_i * (area_i(x) - t)
        # = sum_i lam_i * area_i(x) + t * (1 - sum_i lam_i)
        # For fixed lam, to maximize L w.r.t. t: if sum(lam) > 1, t -> -inf
        # So we need to handle this. Let's use a different formulation.
        
        # Alternative: minimize sum_i lam_i * (t - area_i(x)) + reg * ||x - x0||^2
        # This is equivalent to: maximize sum_i lam_i * area_i(x) - reg * ||x - x0||^2
        # subject to x in triangle
        
        def _primal_objective(params):
            pts = params.reshape(N, 2)
            pts = _project_to_triangle(pts.copy())
            areas = _compute_areas(pts)
            # We want to maximize sum_i lam_i * area_i, so minimize negative
            val = -np.dot(lam, areas)
            # Add regularization to keep points near current
            reg_term = reg * np.sum((pts - current_points)**2)
            return val + reg_term
        
        def _primal_gradient(params):
            pts = params.reshape(N, 2)
            pts = _project_to_triangle(pts.copy())
            grad = np.zeros_like(params)
            
            # Gradient of -sum_i lam_i * area_i
            # area_i = 0.5 * |det_i|
            # d(area_i)/dx depends on sign of det_i
            i_idx = _idx_arr[:, 0]
            j_idx = _idx_arr[:, 1]
            k_idx = _idx_arr[:, 2]
            
            xi, yi = pts[i_idx, 0], pts[i_idx, 1]
            xj, yj = pts[j_idx, 0], pts[j_idx, 1]
            xk, yk = pts[k_idx, 0], pts[k_idx, 1]
            
            det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
            sign = np.sign(det)
            sign[sign == 0] = 1.0
            
            # d(det)/dx_i for each point
            # det = x_i*(y_j - y_k) + x_j*(y_k - y_i) + x_k*(y_i - y_j)
            # d(det)/dx_i = y_j - y_k
            # d(det)/dy_i = -(x_j - x_k)
            # d(det)/dx_j = y_k - y_i
            # d(det)/dy_j = -(x_k - x_i)
            # d(det)/dx_k = y_i - y_j
            # d(det)/dy_k = -(x_i - x_j)
            
            # For area = 0.5 * |det|, d(area)/dx = 0.5 * sign(det) * d(det)/dx
            
            # Accumulate gradients for each point
            for m in range(N):
                # Find all triangles involving point m
                mask_i = (i_idx == m)
                mask_j = (j_idx == m)
                mask_k = (k_idx == m)
                
                # Contribution from being point i
                if np.any(mask_i):
                    lam_i = lam[mask_i]
                    sign_i = sign[mask_i]
                    dy_j_minus_y_k = (yj - yk)[mask_i]
                    dx_j_minus_x_k = (xj - xk)[mask_i]
                    grad[m*2] += 0.5 * np.sum(lam_i * sign_i * dy_j_minus_y_k)
                    grad[m*2+1] += -0.5 * np.sum(lam_i * sign_i * dx_j_minus_x_k)
                
                # Contribution from being point j
                if np.any(mask_j):
                    lam_j = lam[mask_j]
                    sign_j = sign[mask_j]
                    dy_k_minus_y_i = (yk - yi)[mask_j]
                    dx_k_minus_x_i = (xk - xi)[mask_j]
                    grad[m*2] += 0.5 * np.sum(lam_j * sign_j * dy_k_minus_y_i)
                    grad[m*2+1] += -0.5 * np.sum(lam_j * sign_j * dx_k_minus_x_i)
                
                # Contribution from being point k
                if np.any(mask_k):
                    lam_k = lam[mask_k]
                    sign_k = sign[mask_k]
                    dy_i_minus_y_j = (yi - yj)[mask_k]
                    dx_i_minus_x_j = (xi - xj)[mask_k]
                    grad[m*2] += 0.5 * np.sum(lam_k * sign_k * dy_i_minus_y_j)
                    grad[m*2+1] += -0.5 * np.sum(lam_k * sign_k * dx_i_minus_x_j)
            
            # Add regularization gradient
            grad += 2.0 * reg * (pts - current_points).flatten()
            
            return grad
        
        # Use L-BFGS-B for the primal step
        x0 = current_points.flatten()
        try:
            result = minimize(
                _primal_objective,
                x0,
                jac=_primal_gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-6,
                    'ftol': 1e-8
                }
            )
            new_points = result.x.reshape(N, 2)
            new_points = _project_to_triangle(new_points)
        except Exception:
            new_points = current_points
        
        # Evaluate
        areas = _compute_areas(new_points)
        min_a = float(np.min(areas))
        
        if min_a > best_score:
            best_score = min_a
            best_points = new_points.copy()
        
        current_points = new_points
        
        # Dual update: subgradient ascent on lambda
        # dL/dlambda_i = area_i - t where t = min(area)
        # We want to increase lambda_i if area_i < t
        t = min_a
        violations = t - areas  # positive if area_i < t
        violations = np.maximum(violations, 0)
        
        # Normalize violations
        viol_norm = np.sum(violations)
        if viol_norm > 1e-15:
            # Subgradient step
            lam_new = lam + lr * violations / viol_norm
            # Ensure non-negativity and normalize
            lam_new = np.maximum(lam_new, 0)
            lam_sum = np.sum(lam_new)
            if lam_sum > 0:
                lam = lam_new / lam_sum
            else:
                lam = violations / viol_norm
        
        # Decay learning rate
        lr *= 0.95
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    for seed in range(5):
        init_points = _sobol_points_in_triangle(N, seed=seed)
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    grid_points = _initial_guess_grid()
    grid_points = _project_to_triangle(grid_points)
    initializations.append(grid_points)
    
    for seed in range(3):
        np.random.seed(seed + 100)
        bary = np.random.dirichlet([1, 1, 1], size=N)
        u = bary[:, 0]
        v = bary[:, 1]
        init_points = _bary_to_cart(np.column_stack([u, v]))
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    # Optimize each initialization
    for i, init_points in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Nelder-Mead with soft min
        nm_points, nm_score = _nelder_mead_optimize(
            init_points, 
            time_limit=min(12.0, deadline - time.time() - 10),
            beta=50.0
        )
        
        # Local nudge search
        refined_points, refined_score = _local_nudge_search(
            nm_points,
            max_iter=200,
            time_limit=min(8.0, deadline - time.time() - 5)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Phase 2: Lagrangian relaxation on best result
    if best_points is not None and time.time() < deadline - 25:
        lag_points, lag_score = _lagrangian_optimize(
            best_points,
            time_limit=min(20.0, deadline - time.time() - 10),
            max_dual_iters=40
        )
        
        if lag_score > best_score:
            best_score = lag_score
            best_points = lag_points
        
        # Final local nudge on Lagrangian result
        if time.time() < deadline - 8:
            final_points, final_score = _local_nudge_search(
                best_points,
                max_iter=300,
                step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002],
                time_limit=min(8.0, deadline - time.time() - 3)
            )
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
    
    # Fallback
    if best_points is None:
        best_points = _sobol_points_in_triangle(N, seed=0)
        best_points = _project_to_triangle(best_points)
    
    best_points = _project_to_triangle(best_points)
    
    return best_points
