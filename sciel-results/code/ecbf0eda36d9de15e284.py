"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
_N_TRIPLES = len(_IDX_TRIPLES)


def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
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


def _objective(params):
    """Objective: negative min triangle area (for local optimizers)."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)


def _soft_min_objective(params, beta=50.0):
    """
    Smooth soft-min objective for global search.
    Uses log-sum-exp to approximate the minimum of triangle areas smoothly.
    Minimizes -soft_min(areas).
    """
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min via log-sum-exp: soft_min(a) = -1/beta * log(sum(exp(-beta*a)))
    # We want to maximize soft_min, so minimize -soft_min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    return -soft_min_val


def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
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
    
    # Triangular grid patterns
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
    
    # Symmetric: 1 center + rings
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.25
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.5
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    inits.append(np.array(pts))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Differential Evolution (global search)
    # Use soft-min objective for smoothness during DE
    if time_left() > 60.0:
        try:
            de_result = differential_evolution(
                lambda p: _soft_min_objective(p, beta=30.0),
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred',
            )
            if de_result.x is not None:
                # Evaluate with true min area
                true_score = -_objective(de_result.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Also try DE with direct min objective (fewer iterations)
    if time_left() > 40.0:
        try:
            de_result2 = differential_evolution(
                _objective,
                bounds=bounds,
                seed=123,
                maxiter=80,
                popsize=12,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred',
            )
            if de_result2.x is not None:
                true_score = -_objective(de_result2.x)
                if true_score > best_score:
                    best_score = true_score
                    best_params = de_result2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local polish with SLSQP from best
    if best_params is not None and time_left() > 25.0:
        try:
            res = minimize(
                _objective, best_params, method='SLSQP', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Multiple SLSQP restarts from perturbed best
    if best_params is not None and time_left() > 10.0:
        for i in range(8):
            if time_left() < 5.0:
                break
            try:
                noise = rng.normal(0, 0.015, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='SLSQP', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 6: If DE didn't find anything good, try FPS + SLSQP fallback
    if (best_params is None or best_score < 0.001) and time_left() > 20.0:
        for i in range(5):
            if time_left() < 10.0:
                break
            pts = _generate_fps_initialization(rng)
            params = _points_to_param(pts)
            try:
                res = minimize(
                    _objective, params, method='SLSQP', bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        # Fallback
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    best_points = _param_to_points(best_params)
    return best_points
# EVOLVE-BLOCK-END
