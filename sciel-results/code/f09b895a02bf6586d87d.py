import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Triangle vertices
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 0.0])
P2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices for vectorization
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices) # shape (165, 3)

def t_to_points(ts):
    """
    Convert an array of t values (perimeter distances) to an array of Cartesian points.
    ts are in [0, 3).
    """
    ts = np.mod(ts, 3.0)
    points = np.zeros((len(ts), 2))
    
    # P0 -> P1 (t in [0, 1))
    mask0 = (ts >= 0) & (ts < 1)
    if np.any(mask0):
        t0 = ts[mask0]
        points[mask0] = P0 + t0[:, np.newaxis] * (P1 - P0)
        
    # P1 -> P2 (t in [1, 2))
    mask1 = (ts >= 1) & (ts < 2)
    if np.any(mask1):
        t1 = ts[mask1] - 1.0
        points[mask1] = P1 + t1[:, np.newaxis] * (P2 - P1)
        
    # P2 -> P0 (t in [2, 3))
    mask2 = (ts >= 2) & (ts < 3)
    if np.any(mask2):
        t2 = ts[mask2] - 2.0
        points[mask2] = P2 + t2[:, np.newaxis] * (P0 - P2)
        
    return points

def calc_min_area(points):
    """Calculate the minimum area of triangles formed by N points."""
    pi = points[tri_idx[:, 0]]
    pj = points[tri_idx[:, 1]]
    pk = points[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective_boundary(ts):
    """
    Objective function for boundary optimization.
    ts is a flat array of N perimeter parameters.
    """
    points = t_to_points(ts)
    area = calc_min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        score = calc_min_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # 1. Global Search with Differential Evolution on boundary parameters
    bounds = [(0.0, 3.0)] * N
    
    try:
        de_result = differential_evolution(
            objective_boundary,
            bounds,
            seed=42,
            maxiter=150,
            popsize=25,
            tol=1e-7,
            polish=False,
            init='latinhypercube'
        )
        
        de_ts = de_result.x
        de_points = t_to_points(de_ts)
        update_best(de_points)
        
        # Refine the DE result
        if time.time() < deadline - 20:
            try:
                res_refine = minimize(
                    objective_boundary,
                    de_ts,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                refined_ts = res_refine.x
                refined_points = t_to_points(refined_ts)
                update_best(refined_points)
            except Exception:
                pass
                
    except Exception:
        pass

    # 2. Multi-start with structured initializations on boundary
    inits_ts = []
    
    # Evenly spaced
    inits_ts.append(np.linspace(0, 3, N, endpoint=False))
    
    # Random
    rng = np.random.default_rng(123)
    for _ in range(15):
        inits_ts.append(rng.uniform(0, 3, N))
        
    for init_ts in inits_ts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                objective_boundary,
                init_ts,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            refined_points = t_to_points(res.x)
            update_best(refined_points)
        except Exception:
            pass

    # 3. Final Polish
    # If we have a best_points, we don't have its ts directly unless we stored it.
    # We can try to find ts for best_points? No.
    # But the Nelder-Mead steps above are quite thorough.
    
    return best_points

# EVOLVE-BLOCK-END
