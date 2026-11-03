"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples to speed up area calculation
# i < j < k
_IDX_I = None
_IDX_J = None
_IDX_K = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Uses the mapping:
    For each point i, we have params s_i, t_i in [0, 1].
    Barycentric coordinates (relative to V0, V1, V2) are:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    Point = lambda_1 * V1 + lambda_2 * V2 + lambda_3 * V0
    Since V0 is (0,0), Point = lambda_1 * V1 + lambda_2 * V2
    """
    s = params[0::2]
    t = params[1::2]
    
    # Ensure bounds are respected (though optimizer should keep them there)
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    # Cartesian coordinates
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf

    # 1. Global Search using Sobol Sampling
    # We sample in the hypercube [0, 1]^{2N}
    dim = 2 * N
    # Determine number of samples based on time budget
    # Sobol is fast. Let's try 5000 samples first.
    n_samples = 5000
    
    try:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Sobol requires power of 2 for optimal performance, but works otherwise.
        # We round down to nearest power of 2 for efficiency if possible, or just use n.
        # scipy qmc handles non-powers of 2 by truncating.
        samples = sampler.random(n_samples)
        
        # Evaluate all samples
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            if time_left() < 2.0:
                break
            p = samples[i]
            s_val = _objective(p)
            scores[i] = -s_val
            if scores[i] > best_score:
                best_score = scores[i]
                best_params = p.copy()
        
        # If we ran out of time during sampling, return best so far
        if time_left() < 5.0:
            return _param_to_points(best_params)
            
    except Exception:
        # Fallback to random if Sobol fails
        rng = np.random.RandomState(42)
        samples = rng.uniform(0, 1, size=(n_samples, dim))
        for i in range(n_samples):
            if time_left() < 2.0:
                break
            p = samples[i]
            s_val = _objective(p)
            scores[i] = -s_val
            if scores[i] > best_score:
                best_score = scores[i]
                best_params = p.copy()

    if best_params is None:
        # Absolute fallback: center of triangle
        best_params = np.zeros(2 * N)
        # Set to center: s=0.5, t=0.5 -> lam1=0.25, lam2=0.25, lam3=0.5
        # Center of triangle is (1/3, 1/3, 1/3).
        # s(1-t)=1/3, st=1/3, 1-s=1/3 => s=2/3, t=0.5
        # Let's just use a random valid point if nothing else worked
        best_params = rng.uniform(0.1, 0.9, size=2*N)
        best_score = -_objective(best_params)

    # 2. Local Polish with Nelder-Mead
    # We refine the top few candidates from the global search to escape shallow local minima.
    # Get top 5 indices
    top_k = 5
    # Ensure we have enough samples to pick top_k
    valid_indices = np.argsort(-scores)[:min(top_k, n_samples)]
    
    candidates = []
    for idx in valid_indices:
        if scores[idx] > best_score:
             candidates.append(samples[idx])
    
    if not candidates:
        candidates = [best_params]
        
    # Add the global best if not already in top k (it should be)
    if best_params not in candidates:
        candidates.append(best_params)

    for start_p in candidates:
        if time_left() < 10.0:
            break
            
        try:
            res = minimize(
                _objective, start_p, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x
        except Exception:
            pass

    points = _param_to_points(best_params)
    return np.array(points)
# EVOLVE-BLOCK-END
