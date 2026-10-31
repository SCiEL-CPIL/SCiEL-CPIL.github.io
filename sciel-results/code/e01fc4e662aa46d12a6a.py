"""Combined program: Reduced-space Differential Evolution + Multi-start Local Refinement."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """
    Return vector of squared pairwise distances for upper triangle.
    P is (N, 2).
    """
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

def _soft_objective(params, beta):
    """
    Soft min/max surrogate for gradient-based optimization.
    params: flattened array of shape (28,) representing points 2..15.
    Point 0 is (0,0), Point 1 is (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    # Ensure positivity for log
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max via log-sum-exp
    max_d_sq = np.max(d_sq)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min
    min_d_sq = np.min(d_sq)
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min

def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    This aligns the configuration to the reduced space.
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params

def _get_initial_candidates():
    """Generate diverse initial configurations in full space."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    
    # Jittered hex
    for sigma in [0.05, 0.1, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4. Golden angle spiral (Fibonacci)
    n = 16
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n)) / np.sqrt(n) * 2.0
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(spiral)
    candidates.append(spiral + rng.normal(0, 0.1, (16, 2)))
    
    # 5. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 6. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    return candidates

def _local_optimize(params, maxiter_nm=300, maxiter_lbfgs=500):
    """
    Two-phase local optimization:
    1. Nelder-Mead on exact objective
    2. L-BFGS-B on soft objective with annealing
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Nelder-Mead
    try:
        res_nm = minimize(
            _exact_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with annealing on soft objective
    betas = [10.0, 50.0, 200.0, 1000.0]
    for beta in betas:
        try:
            res_lbfgs = minimize(
                lambda p: _soft_objective(p, beta),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            exact_s = _exact_objective(res_lbfgs.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Global Search in Reduced Space using Differential Evolution
    # This is the structural change: DE on 28 dims instead of 32, and more robust than random restarts.
    try:
        bounds = [(-10, 10)] * (28) # 14 points * 2 coords
        
        # We use a moderate popsize and maxiter to ensure it finishes in time.
        # 15 * 28 = 420 individuals. 
        # If maxiter=30, total evals ~ 12600.
        # Each eval is fast (16 points).
        
        res_de = differential_evolution(
            _exact_objective,
            bounds,
            popsize=15,
            maxiter=30,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube',
            updating='deferred'
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_params = res_de.x.copy()
            
    except Exception:
        pass

    elapsed_de = time.time() - start_time
    # print(f"DE took {elapsed_de:.2f}s, score {best_score:.4f}")

    # Phase 2: Local Refinement of the DE result
    if best_params is not None and time.time() < deadline - 10:
        try:
            p_opt, score = _local_optimize(best_params, maxiter_nm=200, maxiter_lbfgs=300)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass

    # Phase 3: Diverse Starts with Local Optimization
    # Use the inspiration's initializers to catch cases where DE missed a good basin.
    candidates = _get_initial_candidates()
    
    for P0 in candidates:
        if time.time() > deadline - 5:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            p_opt, score = _local_optimize(params, maxiter_nm=100, maxiter_lbfgs=200)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass

    # Phase 4: Final High-Precision Polish
    if best_params is not None and time.time() < deadline - 2:
        try:
            # Very high beta for near-exact optimization
            res_final = minimize(
                lambda p: _soft_objective(p, 5000.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass

    # Reconstruct full P
    if best_params is None:
        # Fallback: Grid
        P = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
    else:
        P = np.zeros((N, 2))
        P[0] = [0.0, 0.0]
        P[1] = [1.0, 0.0]
        P[2:] = best_params.reshape(-1, 2)
        
        # Ensure distinctness
        d_sq = _pairwise_dists_sq(P)
        if np.min(d_sq) < 1e-10:
             # Perturb if points are too close
             rng = np.random.default_rng(0)
             P += rng.normal(0, 1e-5, P.shape)

    return P

# EVOLVE-BLOCK-END
