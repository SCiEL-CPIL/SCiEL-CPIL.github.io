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

def perimeter_to_point(t):
    """Convert perimeter parameter t in [0, 3) to a point on the triangle boundary."""
    t = t % 3.0
    if t < 1.0:
        return P0 + t * (P1 - P0)
    elif t < 2.0:
        return P1 + (t - 1.0) * (P2 - P1)
    else:
        return P2 + (t - 2.0) * (P0 - P2)

def perimeter_to_points(ts):
    """Convert array of perimeter parameters to points."""
    return np.array([perimeter_to_point(t) for t in ts])

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

def objective_perimeter(ts):
    """
    Objective function for perimeter optimization.
    ts is an array of N perimeter parameters in [0, 3).
    """
    points = perimeter_to_points(ts)
    area = calc_min_area(points)
    return -area

def generate_regular_perimeter(n):
    """Generate n equally spaced points on the perimeter."""
    ts = np.linspace(0, 3, n, endpoint=False)
    return ts

def refine_perimeter(ts_init, time_limit=5.0):
    """
    Refine perimeter parameters using local optimization.
    """
    start_time = time.time()
    
    # Bounds: each t in [0, 3)
    bounds = [(0.0, 3.0)] * N
    
    try:
        res = minimize(
            objective_perimeter,
            ts_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        if res.success or res.fun < -calc_min_area(perimeter_to_points(ts_init)):
            ts_init = res.x
    except Exception:
        pass
        
    if time.time() - start_time < time_limit:
        try:
            res2 = minimize(
                objective_perimeter,
                ts_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res2.fun < -calc_min_area(perimeter_to_points(ts_init)):
                ts_init = res2.x
        except Exception:
            pass
            
    return ts_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_ts = None
    best_score = -np.inf
    
    def update_best(ts):
        nonlocal best_ts, best_score
        points = perimeter_to_points(ts)
        score = calc_min_area(points)
        if score > best_score:
            best_score = score
            best_ts = ts.copy()
            return True
        return False

    # 1. Global Search with Differential Evolution on perimeter parameters
    bounds = [(0.0, 3.0)] * N
    
    try:
        de_result = differential_evolution(
            objective_perimeter,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            polish=False,
            init='latinhypercube'
        )
        
        update_best(de_result.x)
        
        if time.time() < deadline - 30:
            refined = refine_perimeter(de_result.x, time_limit=15.0)
            update_best(refined)
            
    except Exception:
        pass

    # 2. Multi-start Local Optimization
    inits = []
    
    # Regular spacing
    inits.append(generate_regular_perimeter(N))
    
    # Slightly perturbed regular
    rng = np.random.default_rng(42)
    for _ in range(3):
        base = generate_regular_perimeter(N)
        noise = rng.normal(0, 0.1, N)
        ts = (base + noise) % 3.0
        inits.append(ts)
    
    # Random
    for _ in range(5):
        inits.append(rng.random(N) * 3.0)
    
    # Perturbed best
    if best_ts is not None:
        for _ in range(5):
            noise = rng.normal(0, 0.05, N)
            inits.append((best_ts + noise) % 3.0)

    for init_ts in inits:
        if time.time() > deadline - 15:
            break
        try:
            refined = refine_perimeter(init_ts, time_limit=5.0)
            update_best(refined)
        except Exception:
            pass

    # 3. Final Polish
    if best_ts is not None and time.time() < deadline - 2:
        try:
            bounds = [(0.0, 3.0)] * N
            res_final = minimize(
                objective_perimeter,
                best_ts,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            if res_final.fun < -calc_min_area(perimeter_to_points(best_ts)):
                best_ts = res_final.x
        except Exception:
            pass

    # Convert back to Cartesian coordinates
    if best_ts is not None:
        return perimeter_to_points(best_ts)
    
    # Fallback: regular perimeter
    return perimeter_to_points(generate_regular_perimeter(N))

# EVOLVE-BLOCK-END
