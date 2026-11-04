import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute all triangle indices
indices = list(itertools.combinations(range(N), 3))
n_tri = len(indices)
_IDX_TRIPLES = np.array(indices)
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
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
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _softmin_objective(params, beta=50.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _softmin_grad(params, beta=50.0):
    """Gradient of the soft-min objective."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    dx_i = pj[:, 0] - pi[:, 0]
    dy_i = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    
    cross = dx_i * dy_k - dy_i * dx_k
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    exp_term = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_term)
    
    # d(soft_min)/d(area_m) = -exp_term[m] / sum_exp
    d_softmin_d_area = -exp_term / sum_exp
    
    # d(area_m)/d(point)
    # area = 0.5 * |cross|
    # sign = sign(cross)
    sign = np.sign(cross)
    d_area_d_cross = 0.5 * sign
    
    # d(cross)/d(pi) = (-(dy_k), dx_k)
    # d(cross)/d(pj) = (dy_k, -dx_k)
    # d(cross)/d(pk) = (-dy_i, dx_i)
    
    grad_points = np.zeros((N, 2))
    
    # Contribution from pi
    d_cross_d_pi = np.column_stack([-dy_k, dx_k])
    contrib_i = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pi
    np.add.at(grad_points, _IDX_I, contrib_i)
    
    # Contribution from pj
    d_cross_d_pj = np.column_stack([dy_k, -dx_k])
    contrib_j = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pj
    np.add.at(grad_points, _IDX_J, contrib_j)
    
    # Contribution from pk
    d_cross_d_pk = np.column_stack([-dy_i, dx_i])
    contrib_k = d_softmin_d_area[:, None] * d_area_d_cross[:, None] * d_cross_d_pk
    np.add.at(grad_points, _IDX_K, contrib_k)
    
    # Now map grad_points to grad_params
    # points = [x, y]
    # x = s(1-t)*V1[0] + s*t*V2[0]
    # y = s(1-t)*V1[1] + s*t*V2[1]
    
    s = params[0::2]
    t = params[1::2]
    
    # dx/ds = (1-t)*V1[0] + t*V2[0]
    # dx/dt = -s*V1[0] + s*V2[0]
    # dy/ds = (1-t)*V1[1] + t*V2[1]
    # dy/dt = -s*V1[1] + s*V2[1]
    
    dx_ds = (1.0 - t) * V1[0] + t * V2[0]
    dx_dt = -s * V1[0] + s * V2[0]
    dy_ds = (1.0 - t) * V1[1] + t * V2[1]
    dy_dt = -s * V1[1] + s * V2[1]
    
    grad_s = grad_points[:, 0] * dx_ds + grad_points[:, 1] * dy_ds
    grad_t = grad_points[:, 0] * dx_dt + grad_points[:, 1] * dy_dt
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_s
    grad_params[1::2] = grad_t
    
    # Objective is -soft_min, so gradient is -grad_points mapped to params
    return -grad_params

def _objective_and_grad(params):
    """Objective with gradient for L-BFGS-B."""
    val = _softmin_objective(params, beta=50.0)
    grad = _softmin_grad(params, beta=50.0)
    return val, grad

def _objective_only(params):
    """Objective without gradient."""
    return _softmin_objective(params, beta=50.0)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
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
    
    # Symmetric patterns
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

def _local_search(points, rng, max_iter=200):
    """
    Local search using discrete swaps to improve the minimum triangle area.
    This is a combinatorial exchange heuristic.
    """
    current_points = points.copy()
    current_min_area = _min_triangle_area_vectorized(current_points)
    
    for _ in range(max_iter):
        # Pick a random point to move
        i = rng.randint(N)
        
        # Generate a new candidate position near the current one
        # or a completely random position
        if rng.random() < 0.5:
            # Local perturbation
            new_p = current_points[i] + rng.normal(0, 0.05, 2)
        else:
            # Random position in triangle
            r1, r2 = rng.uniform(0, 1), rng.uniform(0, 1)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            new_p = r1 * V1 + r2 * V2
        
        # Check if new position is inside triangle
        u, v, w = 1 - new_p[0] - 2*new_p[1]/np.sqrt(3), new_p[0] - 0.5*2*new_p[1]/np.sqrt(3), 2*new_p[1]/np.sqrt(3)
        if u < -1e-9 or v < -1e-9 or w < -1e-9:
            continue
        
        # Try the swap
        old_p = current_points[i].copy()
        current_points[i] = new_p
        new_min_area = _min_triangle_area_vectorized(current_points)
        
        if new_min_area > current_min_area:
            current_min_area = new_min_area
        else:
            current_points[i] = old_p
    
    return current_points, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(12345)
    
    # Generate initializations
    inits = []
    
    # Structured inits
    inits.extend(_structured_inits(rng))
    
    # FPS inits
    for _ in range(3):
        inits.append(_generate_fps_initialization(rng))
    
    # Random inits
    for _ in range(5):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * V1 + v * V2
                pts.append(p)
        inits.append(np.array(pts))
    
    # Run optimization from each start
    for i, initial_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        params0 = _points_to_param(initial_pts)
        
        # Try L-BFGS-B with soft-min objective
        try:
            res = minimize(
                _objective_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            params_opt = res.x
            points_opt = _param_to_points(params_opt)
            
            score = _min_triangle_area_vectorized(points_opt)
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                
        except Exception:
            pass
        
        # Try Nelder-Mead from the BFGS result
        if time.time() > deadline - 10:
            break
        
        try:
            res_nm = minimize(
                _objective_only,
                params_opt,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            params_nm = res_nm.x
            points_nm = _param_to_points(params_nm)
            
            score = _min_triangle_area_vectorized(points_nm)
            if score > best_score:
                best_score = score
                best_points = points_nm.copy()
                
        except Exception:
            pass
        
        # Local search refinement
        if time.time() > deadline - 10:
            break
        
        try:
            points_ls, score_ls = _local_search(best_points if best_points is not None else initial_pts, rng, max_iter=100)
            if score_ls > best_score:
                best_score = score_ls
                best_points = points_ls.copy()
        except Exception:
            pass
    
    # Final refinement if we have time
    if best_points is not None and time.time() < deadline - 5:
        params_best = _points_to_param(best_points)
        try:
            res_final = minimize(
                _objective_and_grad,
                params_best,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            points_final = _param_to_points(res_final.x)
            score = _min_triangle_area_vectorized(points_final)
            if score > best_score:
                best_score = score
                best_points = points_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        pts_fallback = [V0, V1, V2]
        M01 = (V0 + V1) / 2
        M12 = (V1 + V2) / 2
        M20 = (V2 + V0) / 2
        pts_fallback.extend([M01, M12, M20])
        C = (V0 + V1 + V2) / 3
        pts_fallback.append(C)
        Q01a = V0 + 0.25 * (V1 - V0)
        Q01b = V0 + 0.75 * (V1 - V0)
        pts_fallback.extend([Q01a, Q01b])
        best_points = np.array(pts_fallback)
    
    return best_points
