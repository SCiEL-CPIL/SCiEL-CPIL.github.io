"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_I = None
_IDX_J = None
_IDX_K = None
_N_TRIPLES = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K, _N_TRIPLES
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
        _N_TRIPLES = len(_IDX_I)
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _squared_areas_objective(params):
    """
    Objective: negative of the minimum squared area (smooth, differentiable).
    We maximize min(areas) by minimizing -min(areas^2).
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    sq_areas = areas ** 2
    return -np.min(sq_areas)

def _soft_min_objective(params, beta=50.0):
    """
    Soft-min objective using log-sum-exp on negative squared areas.
    f(params) = (1/beta) * log(sum(exp(-beta * a_i^2)))
    This approximates min(a_i^2) as beta -> infinity.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    sq_areas = areas ** 2
    # Stable log-sum-exp: log(sum(exp(-beta * x))) = -beta * max(x) + log(sum(exp(-beta * (x - max(x)))))
    neg_beta_x = -beta * sq_areas
    m = np.max(neg_beta_x)
    if m < -700:  # underflow protection
        return -np.min(sq_areas)
    lse = m + np.log(np.sum(np.exp(neg_beta_x - m)))
    return lse / beta

def _soft_min_grad(params, beta=50.0):
    """
    Analytical gradient of the soft-min objective w.r.t. params.
    
    f = (1/beta) * log(sum_i exp(-beta * a_i^2))
    df/da_i^2 = -exp(-beta * a_i^2) / sum_j exp(-beta * a_j^2) = -p_i
    where p_i = exp(-beta * a_i^2) / sum_j exp(-beta * a_j^2)
    
    da_i^2 / dpoint = 2 * a_i * sign(cross_i) * d(cross_i)/dpoint
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    sq_areas = areas ** 2
    
    # Compute softmax weights p_i = exp(-beta * sq_areas_i) / sum_j exp(-beta * sq_areas_j)
    neg_beta_x = -beta * sq_areas
    m = np.max(neg_beta_x)
    if m < -700:
        # Fallback to hard min gradient
        return _squared_areas_grad(params)
    exp_vals = np.exp(neg_beta_x - m)
    p = exp_vals / np.sum(exp_vals)
    
    i_idx, j_idx, k_idx = _get_indices()
    
    # Compute cross products for all triples
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    
    # d(sq_area)/d(cross) = area * sign(cross) = 0.5 * |cross| * sign(cross) = 0.5 * cross
    d_sq_area_d_cross = 0.5 * cross
    
    # Gradient w.r.t. point coordinates
    # cross = (pjx-pix)*(pky-piy) - (pjy-piy)*(pkx-pix)
    d_cross_d_pix = -(pk[:, 1] - pi[:, 1]) + (pj[:, 1] - pi[:, 1])
    d_cross_d_piy = -(pk[:, 0] - pi[:, 0]) + (pj[:, 0] - pi[:, 0])
    
    d_cross_d_pjx = pk[:, 1] - pi[:, 1]
    d_cross_d_pjy = -(pk[:, 0] - pi[:, 0])
    
    d_cross_d_pkx = -(pj[:, 1] - pi[:, 1])
    d_cross_d_pky = pj[:, 0] - pi[:, 0]
    
    # Weight each triple's contribution
    w = p * d_sq_area_d_cross
    
    # Accumulate gradient in point space
    grad_points = np.zeros((N, 2))
    np.add.at(grad_points, (i_idx, 0), -w * d_cross_d_pix)
    np.add.at(grad_points, (i_idx, 1), -w * d_cross_d_piy)
    np.add.at(grad_points, (j_idx, 0), -w * d_cross_d_pjx)
    np.add.at(grad_points, (j_idx, 1), -w * d_cross_d_pjy)
    np.add.at(grad_points, (k_idx, 0), -w * d_cross_d_pkx)
    np.add.at(grad_points, (k_idx, 1), -w * d_cross_d_pky)
    
    # Transform from point space to param space
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return grad_params

def _squared_areas_grad(params):
    """
    Analytical gradient of -min(areas^2) w.r.t. params.
    The gradient is nonzero only for the triple(s) achieving the minimum.
    """
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    sq_areas = areas ** 2
    min_idx = np.argmin(sq_areas)
    
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx[min_idx]]
    pj = points[j_idx[min_idx]]
    pk = points[k_idx[min_idx]]
    
    cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
    area = 0.5 * abs(cross)
    
    # d(areas^2)/d(area) = 2*area, and d(area)/d(cross) = 0.5*sign(cross)
    # So d(areas^2)/d(cross) = area * sign(cross)
    d_sq_area_d_cross = area * np.sign(cross) if abs(cross) > 1e-15 else 0.0
    
    # Gradient w.r.t. point coordinates
    # cross = (pjx-pix)*(pky-piy) - (pjy-piy)*(pkx-pix)
    d_cross_d_pix = -(pk[1] - pi[1]) + (pj[1] - pi[1])
    d_cross_d_piy = -(pk[0] - pi[0]) + (pj[0] - pi[0])
    
    d_cross_d_pjx = pk[1] - pi[1]
    d_cross_d_pjy = -(pk[0] - pi[0])
    
    d_cross_d_pkx = -(pj[1] - pi[1])
    d_cross_d_pky = pj[0] - pi[0]
    
    # Gradient of objective = -min(sq_areas) w.r.t. points
    # = -2 * area * sign(cross) * d_cross/d_point for the min triple
    grad_points = np.zeros((N, 2))
    w = -d_sq_area_d_cross
    
    grad_points[i_idx[min_idx]] += w * np.array([d_cross_d_pix, d_cross_d_piy])
    grad_points[j_idx[min_idx]] += w * np.array([d_cross_d_pjx, d_cross_d_pjy])
    grad_points[k_idx[min_idx]] += w * np.array([d_cross_d_pkx, d_cross_d_pky])
    
    # Transform from point space to param space
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    
    dx_ds = (1 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_params[1::2] = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    return grad_params

def _qp_trust_constr(params, rng, time_budget=30.0):
    """
    Use scipy trust-constr with soft-min objective and beta annealing.
    """
    from scipy.optimize import minimize
    
    t_start = time.time()
    bounds = [(0.0, 1.0)] * (2 * N)
    
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    # Beta annealing schedule: start with small beta (smooth), increase to large (sharp)
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() - t_start > time_budget:
            break
        
        # Define objective and gradient with current beta
        def obj(params, beta=beta):
            return _soft_min_objective(params, beta)
        
        def grad(params, beta=beta):
            return _soft_min_grad(params, beta)
        
        try:
            res = minimize(
                obj,
                current_params,
                jac=grad,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
            )
            score_res = -_objective(res.x)
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
                current_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with hard-min objective
    if time.time() - t_start < time_budget - 5.0:
        try:
            res = minimize(
                _squared_areas_objective,
                best_params,
                jac=_squared_areas_grad,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 300, 'gtol': 1e-10, 'xtol': 1e-12, 'verbose': 0}
            )
            score_res = -_objective(res.x)
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
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
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
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

def _simulated_annealing(params, rng, max_iter=5000, t_start=0.1, t_end=0.001, deadline=150.0):
    """
    Simulated annealing to explore the parameter space.
    """
    current_params = params.copy()
    current_score = -_objective(current_params)
    best_params_sa = current_params.copy()
    best_score_sa = current_score
    
    n_steps = max_iter
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, n_steps - 1))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = t_start * (cooling_rate ** step)
        
        # Generate neighbor: mix of jitter and swap
        r = rng.random()
        if r < 0.3:
            # Topological swap: swap barycentric coords of two random points
            i, j = rng.choice(N, 2, replace=False)
            new_params = current_params.copy()
            # Swap s and t for points i and j
            new_params[i*2], new_params[j*2] = new_params[j*2], new_params[i*2]
            new_params[i*2+1], new_params[j*2+1] = new_params[j*2+1], new_params[i*2+1]
        elif r < 0.7:
            # Local jitter: Gaussian perturbation
            noise = rng.normal(0, 0.05, size=2*N)
            new_params = np.clip(current_params + noise, 0.0, 1.0)
        else:
            # Larger move: perturb one point significantly
            i = rng.randint(N)
            new_params = current_params.copy()
            new_params[i*2] = np.clip(current_params[i*2] + rng.normal(0, 0.2), 0.0, 1.0)
            new_params[i*2+1] = np.clip(current_params[i*2+1] + rng.normal(0, 0.2), 0.0, 1.0)
        
        new_score = -_objective(new_params)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_params = new_params
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_params_sa = current_params.copy()
    
    return best_params_sa, best_score_sa

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
    n_fps_inits = 10
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
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Simulated annealing to escape local minima
    if best_params is not None and time_left() > 50.0:
        try:
            sa_params, sa_score = _simulated_annealing(best_params, rng, max_iter=3000, deadline=deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_params = sa_params.copy()
        except Exception:
            pass

    # 4. Polish the best solution found with Nelder-Mead
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 5. Try a few more SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 25.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass

    # 6. Use trust-constr with soft-min objective and beta annealing
    if best_params is not None and time_left() > 40.0:
        try:
            qp_params, qp_score = _qp_trust_constr(best_params, rng, time_budget=min(35.0, time_left() - 25.0))
            true_score = -_objective(qp_params)
            if true_score > best_score:
                best_score = true_score
                best_params = qp_params.copy()
        except Exception:
            pass

    # 7. Final high-precision polish with SLSQP then Nelder-Mead
    if best_params is not None and time_left() > 15.0:
        # First SLSQP polish
        try:
            res = minimize(
                _objective, best_params, method='SLSQP', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'eps': 1e-12}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Then Nelder-Mead with very tight tolerances
        if time_left() > 10.0:
            try:
                res = minimize(
                    _objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
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
    
    # Validate all points are inside the triangle (with small tolerance)
    h = np.sqrt(3) / 2
    for i in range(N):
        x, y = best_points[i]
        # Check barycentric coordinates
        lam2 = 2.0 * y / np.sqrt(3)
        lam1 = x - 0.5 * lam2
        lam3 = 1.0 - lam1 - lam2
        # If any coordinate is significantly negative, project to nearest point
        if lam1 < -1e-8 or lam2 < -1e-8 or lam3 < -1e-8:
            # Project to triangle: clip barycentric coords
            lams = np.array([lam1, lam2, lam3])
            lams = np.maximum(lams, 0.0)
            lams = lams / np.sum(lams)
            best_points[i] = lams[0] * V0 + lams[1] * V1 + lams[2] * V2
    
    return best_points
