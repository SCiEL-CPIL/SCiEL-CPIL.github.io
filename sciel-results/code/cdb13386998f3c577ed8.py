import numpy as np
import time
import itertools
from scipy.optimize import minimize, basinhopping

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_spread_objective(params, beta=10.0):
    """
    Smooth surrogate objective using soft max/min with temperature beta.
    params: flattened array of shape (2*(N-2),) representing points 2..N-1.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    
    # Soft max: (1/beta) * log(sum(exp(beta * d_sq)))
    # Soft min: - (1/beta) * log(sum(exp(-beta * d_sq)))
    
    # For numerical stability, shift by max
    max_d = np.max(d_sq)
    min_d = np.min(d_sq)
    
    # Soft max
    exp_max = np.exp(beta * (d_sq - max_d))
    soft_max = max_d + np.log(np.sum(exp_max)) / beta
    
    # Soft min
    exp_min = np.exp(-beta * (d_sq - min_d))
    soft_min = min_d - np.log(np.sum(exp_min)) / beta
    
    if soft_min < 1e-6:
        return 1e10
    
    return soft_max / soft_min

def _exact_objective(params):
    """
    Exact objective for local refinement.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Hexagonal Lattice Subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    d_sqs = np.sum(hex_points**2, axis=1)
    
    outer_idx = np.where(np.isclose(d_sqs, 4.0))[0]
    inner_idx = np.where(np.isclose(d_sqs, 3.0))[0]
    mid_idx = np.where(np.isclose(d_sqs, 1.0))[0]
    
    rng = np.random.default_rng(42)
    
    # Try removing different combinations
    for _ in range(10):
        if len(outer_idx) >= 3:
            remove_idx = rng.choice(outer_idx, 3, replace=False)
        elif len(outer_idx) >= 2 and len(inner_idx) >= 1:
            remove_idx = np.concatenate([rng.choice(outer_idx, 2, replace=False), 
                                         rng.choice(inner_idx, 1, replace=False)])
        else:
            continue
        keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
        if len(keep_idx) == N:
            candidates.append(hex_points[keep_idx])
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Random Perturbed Grid
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    # 5. Random Gaussian Cluster
    rng2 = np.random.default_rng(123)
    gauss = rng2.normal(0, 1, (N, 2))
    candidates.append(gauss)
    
    # 6. Ring of 16
    pts_ring = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_ring.append([2 * np.cos(ang), 2 * np.sin(ang)])
    candidates.append(np.array(pts_ring))
    
    # 7. Fibonacci spiral
    pts_spiral = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(N):
        r = np.sqrt(k + 1) * 0.5
        theta = k * golden_angle
        pts_spiral.append([r * np.cos(theta), r * np.sin(theta)])
    candidates.append(np.array(pts_spiral))
    
    return candidates

def _prepare_for_optimization(P):
    """
    Fix P[0]=(0,0) and P[1]=(1,0) by finding the min-distance pair in P,
    rotating/ translating/scaling the whole set so that pair becomes (0,0) and (1,0).
    Returns the optimized parameter vector (coords of P[2:] to P[15]).
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    p, q = np.unravel_index(min_idx, (N, N))
    if p > q:
        p, q = q, p
    
    p0 = P[p]
    p1 = P[q]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return None
        
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = p
    perm[1] = q
    remaining = [idx for idx in range(N) if idx != p and idx != q]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    return init_params

def _local_optimize(params, start_time, deadline, beta=10.0):
    """
    Run local optimization with multiple methods and return best params.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Check time
    if time.time() > deadline:
        return best_params, best_score
    
    # Phase 1: L-BFGS-B on soft objective
    try:
        res = minimize(
            _soft_spread_objective,
            best_params,
            method='L-BFGS-B',
            args=(beta,),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline and res.fun < best_score:
            # Verify with exact objective
            exact_s = _exact_objective(res.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead on exact objective
    if time.time() < deadline:
        try:
            res2 = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Phase 3: Another L-BFGS-B pass with higher beta
    if time.time() < deadline:
        try:
            res3 = minimize(
                _soft_spread_objective,
                best_params,
                method='L-BFGS-B',
                args=(20.0,),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res3.fun < best_score:
                exact_s = _exact_objective(res3.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = res3.x
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    candidates = _get_initial_candidates()
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Evaluate initial candidates and find best starting point
    for P0 in candidates:
        if time.time() > deadline:
            break
        init_params = _prepare_for_optimization(P0)
        if init_params is None:
            continue
        
        score = _exact_objective(init_params)
        if score < best_score:
            best_score = score
            best_params = init_params

    # Phase 2: Global Search with Basinhopping from best candidate
    if best_params is not None and time.time() < deadline - 20:
        def callback(x, f, accept):
            nonlocal best_score
            if f < best_score:
                best_score = f
                best_params = x.copy()
        
        try:
            bh_res = basinhopping(
                _soft_spread_objective,
                best_params,
                niter=30,
                minimizer_kwargs={'method': 'L-BFGS-B', 'args': (10.0,), 
                                  'options': {'maxiter': 50, 'ftol': 1e-8}},
                callback=callback,
                T=1.0,
                stepsize=0.5
            )
        except Exception:
            pass
    
    # Phase 3: Local Refinement with multiple passes
    if best_params is not None and time.time() < deadline - 5:
        best_params, best_score = _local_optimize(best_params, start_time, deadline)
    
    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    # Construct final points
    if best_params is not None:
        final_pts = np.zeros((N, 2))
        final_pts[0] = [0.0, 0.0]
        final_pts[1] = [1.0, 0.0]
        final_pts[2:] = best_params.reshape(-1, 2)
        
        # Check validity
        if _score(final_pts) > 1e6:
            return candidates[0]
            
        return final_pts
    else:
        return candidates[0]

# EVOLVE-BLOCK-END
