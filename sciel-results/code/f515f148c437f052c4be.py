"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

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

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    # Handle singularity at 0,0
    u = bary[:, 0]
    v = bary[:, 1]
    s = u + v
    # If s is very small, leave as 0
    # If s > 1, project
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    # Clip small negatives to 0
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _objective(x):
    """Objective: negative min triangle area."""
    bary = x.reshape(N, 2)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    best_x = None
    best_score = -np.inf

    def time_left():
        return deadline - time.time()

    # 1. Global Search using Differential Evolution
    # Bounds for barycentric coordinates u, v in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    try:
        # Use sobol init if available, else latinhypercube
        try:
            res_de = differential_evolution(
                _objective, bounds,
                popsize=15,
                maxiter=100,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol'
            )
        except ValueError:
            # Fallback for older scipy
            res_de = differential_evolution(
                _objective, bounds,
                popsize=15,
                maxiter=100,
                tol=1e-8,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
        
        best_x = res_de.x
        best_score = -res_de.fun
        
        if time_left() < 10.0:
            # Not enough time for polish
            points = _bary_to_cart(best_x.reshape(N, 2))
            return points
            
    except Exception:
        # Fallback to random initialization if DE fails
        rng = np.random.RandomState(42)
        best_x = rng.uniform(0, 1, size=2 * N)
        best_score = -_objective(best_x)

    # 2. Local Polish with Powell's Method
    # Powell is robust for non-smooth functions and low dimensions
    try:
        res_powell = minimize(
            _objective, best_x, method='Powell',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        score_powell = -res_powell.fun
        if score_powell > best_score:
            best_x = res_powell.x
            best_score = score_powell
            
        if time_left() < 5.0:
            points = _bary_to_cart(best_x.reshape(N, 2))
            return points
            
    except Exception:
        pass

    # 3. Final Polish with Nelder-Mead for high precision
    try:
        res_nm = minimize(
            _objective, best_x, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        score_nm = -res_nm.fun
        if score_nm > best_score:
            best_x = res_nm.x
            best_score = score_nm
    except Exception:
        pass

    points = _bary_to_cart(best_x.reshape(N, 2))
    return np.array(points)
# EVOLVE-BLOCK-END
