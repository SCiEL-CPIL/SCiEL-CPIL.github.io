"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Uses soft-min surrogate with L-BFGS-B gradient optimization, simulated annealing,
and active-set LP polishing for the piecewise-linear structure.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linprog

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = len(_IDX_TRIPLES)


def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _softmin_objective(params, beta=200.0):
    """Soft-min surrogate: -log(sum(exp(-beta * area_i))) / beta.
    This is smooth and differentiable, approximating the min as beta -> inf."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    # Shift for numerical stability
    a_min = np.min(areas)
    shifted = areas - a_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return -(a_min - log_sum / beta)


def _softmin_grad(params, beta=200.0):
    """Gradient of the soft-min objective with respect to params."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    # Cross products: cross = (pj - pi) x (pk - pi)
    dx_j = pj[:, 0] - pi[:, 0]
    dy_j = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    cross = dx_j * dy_k - dy_j * dx_k
    areas = 0.5 * np.abs(cross)
    
    # Sign of cross product (for abs derivative)
    sign = np.sign(cross)
    sign[sign == 0] = 1.0
    
    # Soft-min weights
    a_min = np.min(areas)
    shifted = areas - a_min
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of area w.r.t. each point's coordinates
    # d(area)/d(pi) = -0.5 * sign * (dy_k, -dx_k)  [from chain rule on abs]
    # d(area)/d(pj) =  0.5 * sign * (dy_k, -dx_k)
    # d(area)/d(pk) = -0.5 * sign * (dy_j, -dx_j)
    
    # Build gradient w.r.t. points (2N values)
    grad_pts = np.zeros((N, 2))
    
    # Contribution from pi (point i)
    # d(area_i)/d(xi) = -0.5 * sign_i * dy_k_i
    # d(area_i)/d(yi) =  0.5 * sign_i * dx_k_i
    g_ix = -0.5 * sign * weights * dy_k
    g_iy =  0.5 * sign * weights * dx_k
    
    # Contribution from pj (point j)
    # d(area_i)/d(xj) =  0.5 * sign_i * dy_k_i
    # d(area_i)/d(yj) = -0.5 * sign_i * dx_k_i
    g_jx =  0.5 * sign * weights * dy_k
    g_jy = -0.5 * sign * weights * dx_k
    
    # Contribution from pk (point k)
    # d(area_i)/d(xk) = -0.5 * sign_i * dy_j_i
    # d(area_i)/d(yk) =  0.5 * sign_i * dx_j_i
    g_kx = -0.5 * sign * weights * dy_j
    g_ky =  0.5 * sign * weights * dx_j
    
    # Accumulate using np.add.at
    np.add.at(grad_pts, _IDX_I, np.column_stack([g_ix, g_iy]))
    np.add.at(grad_pts, _IDX_J, np.column_stack([g_jx, g_jy]))
    np.add.at(grad_pts, _IDX_K, np.column_stack([g_kx, g_ky]))
    
    # Chain rule: params -> points
    # s = params[0::2], t = params[1::2]
    # point = s*(1-t)*V1 + s*t*V2
    # dx/ds = (1-t)*V1[0] + t*V2[0]
    # dx/dt = -s*V1[0] + s*V2[0]
    # dy/ds = (1-t)*V1[1] + t*V2[1]
    # dy/dt = -s*V1[1] + s*V2[1]
    s = params[0::2]
    t = params[1::2]
    
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_pts[:, 0] * dx_ds + grad_pts[:, 1] * dy_ds
    grad_params[1::2] = grad_pts[:, 0] * dx_dt + grad_pts[:, 1] * dy_dt
    
    return grad_params


def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _param_to_points(params):
    """Map parameters in [0, 1]^{2N} to points inside the triangle using barycentric coords."""
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])


def _points_to_param(points):
    """Map points in the triangle to parameters (s, t)."""
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params


def _hard_objective(params):
    """Hard objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)


def _score_params(params):
    """Score a parameter configuration (higher is better)."""
    points = _param_to_points(params)
    return _min_triangle_area_vectorized(points)


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    # Clip x to [0, 1]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    # Clip y to [0, H]
    pts[:, 1] = np.clip(pts[:, 1], 0.0, H)
    # For each point, if y > H * (1 - x) or y > H * x (when x > 0.5), clamp
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        if y > H:
            y = H
        if x <= 0.5:
            max_y = H * x / 0.5
        else:
            max_y = H * (1 - x) / 0.5
        if y > max_y:
            y = max_y
        pts[i] = [x, y]
    return pts


def _slsqp_optimize(params, maxiter=200, ftol=1e-9):
    """Optimize using SLSQP with hard objective."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _hard_objective, params, method='SLSQP', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        return res.x
    except Exception:
        return params


def _nelder_mead_polish(params, maxiter=5000):
    """Polish using Nelder-Mead."""
    try:
        res = minimize(
            _hard_objective, params, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params


def _l_bfgs_softmin(params, beta=200.0, maxiter=500):
    """Optimize using L-BFGS-B on the soft-min surrogate."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _softmin_objective, params, method='L-BFGS-B',
            jac=lambda p: _softmin_grad(p, beta),
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return params


def _active_set_lp(points, n_active=50, max_iter=20):
    """Active-set LP: enforce the k smallest triangle areas as constraints.
    Maximizes t subject to area_i >= t for the active set, points in triangle.
    
    Variables: 2N point coords + 1 (t) = 2N+1 variables.
    Constraints: area_i >= t for each active triple, and triangle containment.
    """
    # Compute all areas
    all_areas = _all_triangle_areas(points)
    # Get indices of smallest areas
    n_active_actual = min(n_active, N_TRIPLES)
    active_idx = np.argsort(all_areas)[:n_active_actual]
    
    # Variables: x0, y0, x1, y1, ..., x10, y10, t
    n_vars = 2 * N + 1
    
    # Objective: maximize t -> minimize -t
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -t
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each active triple (i,j,k): area >= t
    # area = 0.5 * |cross| >= t
    # This is non-linear due to abs. But for a fixed orientation, we can linearize.
    # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # area = 0.5 * cross (assuming positive orientation)
    # 0.5 * cross >= t  =>  cross - 2t >= 0  =>  -cross + 2t <= 0
    
    # We need to handle the sign. For each active triple, determine sign from current config.
    A_rows = []
    b_vals = []
    
    for idx in active_idx:
        i, j, k = _IDX_TRIPLES[idx]
        # Compute cross sign at current points
        pi = points[i]
        pj = points[j]
        pk = points[k]
        cross_val = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        sgn = 1.0 if cross_val >= 0 else -1.0
        
        # cross = sgn * (0.5 * |cross|) ... actually area = 0.5*|cross|
        # We want: 0.5 * sgn * cross >= t  (since sgn*cross = |cross|)
        # sgn * cross - 2t >= 0
        # -sgn * cross + 2t <= 0
        
        # cross = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
        # = xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi
        # This is bilinear (product of variables), so LP won't work directly.
        # 
        # Alternative: use the fact that area = 0.5 * |cross| and for small perturbations
        # around current config, the sign is fixed. We can linearize:
        # area ≈ area_current + grad_area · (delta_points)
        # This gives a linear constraint.
        
        # Gradient of area w.r.t. points (same as in softmin_grad but for single area)
        # d(area)/d(xi) = -0.5 * sgn * (dy_k, -dx_k)
        # etc.
        dx_j = pj[0] - pi[0]
        dy_j = pj[1] - pi[1]
        dx_k = pk[0] - pi[0]
        dy_k = pk[1] - pi[1]
        
        g_ix = -0.5 * sgn * dy_k
        g_iy =  0.5 * sgn * dx_k
        g_jx =  0.5 * sgn * dy_k
        g_jy = -0.5 * sgn * dx_k
        g_kx = -0.5 * sgn * dy_j
        g_ky =  0.5 * sgn * dx_j
        
        # Linearized: area_current + g_i·dpi + g_j·dpj + g_k·dpk >= t
        # => g_i·dpi + g_j·dpj + g_k·dpk - t >= -(area_current)
        # => -g_i·dpi - g_j·dpj - g_k·dpk + t <= area_current
        
        row = np.zeros(n_vars)
        row[2*i]   = -g_ix
        row[2*i+1] = -g_iy
        row[2*j]   = -g_jx
        row[2*j+1] = -g_jy
        row[2*k]   = -g_kx
        row[2*k+1] = -g_ky
        row[-1]    = 1.0
        
        A_rows.append(row)
        b_vals.append(all_areas[idx])
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Triangle containment constraints (linear):
    # y >= 0
    # y <= H * x  (for x <= 0.5, left edge)
    # y <= H * (1-x)  (for x >= 0.5, right edge)
    # x >= 0, x <= 1
    # Actually the triangle is: y >= 0, y <= H*x (left side from (0,0) to (0.5,H)),
    # y <= H*(1-x) (right side from (1,0) to (0.5,H))
    
    # Add: -y_i <= 0 for all i
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i+1] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # x_i >= 0: -x_i <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # x_i <= 1: x_i <= 1
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = 1.0
        A_rows.append(row)
        b_vals.append(1.0)
    
    # y_i <= H * x_i:  -H*x_i + y_i <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = -H
        row[2*i+1] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    # y_i <= H * (1 - x_i):  H*x_i + y_i <= H
    for i in range(N):
        row = np.zeros(n_vars)
        row[2*i] = H
        row[2*i+1] = 1.0
        A_rows.append(row)
        b_vals.append(H)
    
    # t >= 0
    row = np.zeros(n_vars)
    row[-1] = -1.0
    A_rows.append(row)
    b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds for variables
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    for i in range(N):
        ub[2*i] = 1.0
        ub[2*i+1] = H
    ub[-1] = 1.0  # t <= 1 (triangle area)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(lb, ub), method='highs')
        if result.success:
            new_points = result.x[:2*N].reshape(N, 2)
            # Clamp to triangle
            new_points = _clamp_to_triangle(new_points)
            return new_points
    except Exception:
        pass
    
    return points


def _simulated_annealing(points, start_temp=0.5, end_temp=0.01, n_iter=5000, rng=None):
    """Simulated annealing with topological swaps and local jitter.
    
    Two types of moves:
    1. Local jitter: perturb one or two points slightly
    2. Topological swap: swap two points and then optimize locally
    
    Returns improved points and score.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current = points.copy()
    current_score = _min_triangle_area_vectorized(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(n_iter):
        # Cooling schedule
        alpha = it / max(n_iter - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** alpha
        
        move_type = rng.random()
        if move_type < 0.6:
            # Local jitter: perturb 1-3 random points
            n_perturb = rng.randint(1, 4)
            idx = rng.choice(N, n_perturb, replace=False)
            new_points = current.copy()
            for i in idx:
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
                new_points[i, 0] += dx
                new_points[i, 1] += dy
            new_points = _clamp_to_triangle(new_points)
        elif move_type < 0.85:
            # Swap two points and jitter both
            i, j = rng.choice(N, 2, replace=False)
            new_points = current.copy()
            new_points[i], new_points[j] = new_points[j].copy(), new_points[i].copy()
            # Add small jitter to break symmetry
            for k in [i, j]:
                dx = rng.normal(0, 0.03)
                dy = rng.normal(0, 0.03)
                new_points[k, 0] += dx
                new_points[k, 1] += dy
            new_points = _clamp_to_triangle(new_points)
        else:
            # Global move: replace 1-2 points with new random positions
            n_replace = rng.randint(1, 3)
            idx = rng.choice(N, n_replace, replace=False)
            new_points = current.copy()
            for i in idx:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                new_points[i] = r1 * V1 + r2 * V2
            new_points = _clamp_to_triangle(new_points)
        
        new_score = _min_triangle_area_vectorized(new_points)
        
        # Accept or reject
        if new_score >= current_score:
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best = new_points.copy()
                best_score = new_score
        else:
            delta = current_score - new_score
            if delta < 1e-15:
                accept = True
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-15))
                accept = rng.random() < accept_prob
            if accept:
                current = new_points
                current_score = new_score
    
    return best, best_score


def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """Generate an initial configuration using Greedy Farthest Point Sampling."""
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points


def _structured_inits(rng):
    """Generate structured initializations based on grid or symmetric patterns."""
    inits = []
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    # Symmetric patterns: 1 center + 10 around
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    inits.append(np.array(pts))
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 12
    for i in range(n_fps_inits):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(10):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.05, 0.95, size=N)
        t = rng.uniform(0.05, 0.95, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    # 2. Optimize each initialization with L-BFGS-B on soft-min (fast, smooth)
    for start_p in initial_params:
        if time_left() < 25.0:
            break
            
        try:
            # First pass: moderate beta
            refined = _l_bfgs_softmin(start_p, beta=50.0, maxiter=200)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 3. Refine best with higher beta (closer to true min)
    if best_params is not None and time_left() > 30.0:
        try:
            refined = _l_bfgs_softmin(best_params, beta=200.0, maxiter=300)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 4. Run simulated annealing on the best solution found so far
    if best_params is not None and time_left() > 45.0:
        try:
            best_points = _param_to_points(best_params)
            sa_points, sa_score = _simulated_annealing(
                best_points, start_temp=0.3, end_temp=0.005, 
                n_iter=4000, rng=rng
            )
            if sa_score > best_score:
                best_score = sa_score
                best_params = _points_to_param(sa_points)
        except Exception:
            pass

    # 5. Polish with L-BFGS-B after SA
    if best_params is not None and time_left() > 30.0:
        try:
            refined = _l_bfgs_softmin(best_params, beta=200.0, maxiter=300)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 6. Active-set LP polish
    if best_params is not None and time_left() > 25.0:
        try:
            best_points = _param_to_points(best_params)
            lp_points = _active_set_lp(best_points, n_active=60, max_iter=10)
            lp_score = _min_triangle_area_vectorized(lp_points)
            if lp_score > best_score:
                best_score = lp_score
                best_params = _points_to_param(lp_points)
        except Exception:
            pass

    # 7. Nelder-Mead polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = _nelder_mead_polish(best_params, maxiter=3000)
            score_res = _score_params(res)
            if score_res > best_score:
                best_score = score_res
                best_params = res.copy()
        except Exception:
            pass

    # 8. Try more SA runs from perturbed best solution
    if best_params is not None and time_left() > 35.0:
        for i in range(4):
            if time_left() < 20.0:
                break
            try:
                best_points = _param_to_points(best_params)
                noise = rng.normal(0, 0.03, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                pert_points = _param_to_points(perturbed)
                sa_points, sa_score = _simulated_annealing(
                    pert_points, start_temp=0.2, end_temp=0.005,
                    n_iter=2500, rng=rng
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_params = _points_to_param(sa_points)
                    # Quick L-BFGS polish after each improvement
                    if time_left() > 20.0:
                        refined = _l_bfgs_softmin(best_params, beta=200.0, maxiter=200)
                        score_res = _score_params(refined)
                        if score_res > best_score:
                            best_score = score_res
                            best_params = refined.copy()
            except Exception:
                pass

    # 9. Final high-precision polish with SLSQP
    if best_params is not None and time_left() > 12.0:
        try:
            res = _slsqp_optimize(best_params, maxiter=300)
            score_res = _score_params(res)
            if score_res > best_score:
                best_score = score_res
                best_params = res.copy()
        except Exception:
            pass

    # 10. Final active-set LP
    if best_params is not None and time_left() > 8.0:
        try:
            best_points = _param_to_points(best_params)
            lp_points = _active_set_lp(best_points, n_active=80, max_iter=5)
            lp_score = _min_triangle_area_vectorized(lp_points)
            if lp_score > best_score:
                best_score = lp_score
                best_params = _points_to_param(lp_points)
        except Exception:
            pass

    if best_params is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    # Convert best params to points
    best_points = _param_to_points(best_params)
    
    return best_points
