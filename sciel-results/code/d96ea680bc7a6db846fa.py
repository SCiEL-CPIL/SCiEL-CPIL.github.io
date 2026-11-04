"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Combines SLSQP/Nelder-Mead from current program with soft-min surrogate and DE from inspiration."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

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
    if len(areas) == 0:
        return 0.0
    return np.min(areas)


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


def _softmin_objective(params, beta=50.0):
    """Differentiable soft-min objective: negative soft-min of triangle areas."""
    points = _param_to_points(params)
    areas = _all_triangle_areas(points)
    # Soft min: -log-sum-exp(-beta * areas) / beta
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum / beta
    return -soft_min


def _hard_objective(params):
    """Hard objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)


def _score_params(params):
    """Score a parameter configuration (higher is better)."""
    points = _param_to_points(params)
    return _min_triangle_area_vectorized(points)


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
    """Polish using Nelder-Mead with adaptive settings."""
    try:
        res = minimize(
            _hard_objective, params, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params


def _l_bfgs_softmin(params, beta=50.0, maxiter=200):
    """Optimize using L-BFGS-B with soft-min objective."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _softmin_objective, params, args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return params


def _beta_anneal_refine(params, betas=(20.0, 40.0, 80.0, 160.0, 320.0)):
    """Refine with increasing beta values, then hard polish."""
    current = params
    for beta in betas:
        current = _l_bfgs_softmin(current, beta=beta, maxiter=100)
    current = _nelder_mead_polish(current, maxiter=3000)
    return current


def _differential_evolution_global(rng, maxiter=150, popsize=12):
    """Global search using differential evolution."""
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def de_objective(x):
        return _hard_objective(x)
    
    try:
        result = differential_evolution(
            de_objective, bounds,
            maxiter=maxiter, popsize=popsize,
            seed=int(rng.randint(0, 2**31)),
            tol=1e-10, atol=1e-12,
            mutation=(0.5, 1.5), recombination=0.9,
            init='latinhypercube'
        )
        return result.x
    except Exception:
        return None


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
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 35.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 35.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(8):
        if time_left() < 35.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    # 2. Optimize each initialization: soft-min L-BFGS-B first, then SLSQP
    for start_p in initial_params:
        if time_left() < 15.0:
            break
            
        try:
            # Soft-min optimization with moderate beta for smooth gradients
            refined = _l_bfgs_softmin(start_p, beta=40.0, maxiter=100)
            
            # Then SLSQP with hard objective for precise constraint handling
            refined2 = _slsqp_optimize(refined, maxiter=150)
            
            score_res = _score_params(refined2)
            if score_res > best_score:
                best_score = score_res
                best_params = refined2.copy()
        except Exception:
            pass

    # 3. Polish the best solution with beta annealing + Nelder-Mead
    if best_params is not None and time_left() > 35.0:
        try:
            refined = _beta_anneal_refine(best_params)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 4. Differential evolution global search
    if time_left() > 40.0:
        try:
            de_result = _differential_evolution_global(rng, maxiter=100, popsize=10)
            if de_result is not None:
                score_res = _score_params(de_result)
                if score_res > best_score:
                    best_score = score_res
                    best_params = de_result.copy()
        except Exception:
            pass

    # 5. Try a few more SLSQP runs from perturbed best solution
    if best_params is not None and time_left() > 25.0:
        for i in range(5):
            if time_left() < 12.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = _slsqp_optimize(perturbed, maxiter=100)
                score_res = _score_params(res)
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.copy()
            except Exception:
                pass

    # 6. Final high-precision polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = _nelder_mead_polish(best_params, maxiter=5000)
            score_res = _score_params(res)
            if score_res > best_score:
                best_score = score_res
                best_params = res.copy()
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
