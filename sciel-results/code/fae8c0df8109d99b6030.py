import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2]
])

AREA_TRI = np.sqrt(3) / 4.0

def bary_to_cartesian(params):
    """
    params: array of shape (N, 2) containing (u, v) for each point.
    Returns: array of shape (N, 2) containing (x, y).
    """
    u = params[:, 0]
    v = params[:, 1]
    w = 1.0 - u - v
    points = u[:, None] * V[1] + v[:, None] * V[2] + w[:, None] * V[0]
    return points

def min_triangle_area_cartesian(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def objective_bary(params):
    """
    Objective function to minimize: -min_area.
    params: flat array of shape (2N,).
    """
    p = params.reshape(N, 2)
    # Check validity (should be guaranteed by constraints, but for safety)
    # If u+v > 1, SLSQP might evaluate it. We return a large penalty.
    if np.any(p[:, 0] + p[:, 1] > 1.0 + 1e-9) or np.any(p < -1e-9):
        return 1e6
    points = bary_to_cartesian(p)
    return -min_triangle_area_cartesian(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_params = None
    best_score = 0.0
    
    # Constraints for SLSQP
    # u_i >= 0, v_i >= 0, u_i + v_i <= 1
    # SLSQP constraints are of the form: fun(x) >= 0 (ineq) or = 0 (eq)
    
    # Inequalities:
    # 1. u_i >= 0
    # 2. v_i >= 0
    # 3. 1 - u_i - v_i >= 0
    
    # We can define a single constraint function that returns a vector.
    def constraints(params):
        p = params.reshape(N, 2)
        u = p[:, 0]
        v = p[:, 1]
        # u >= 0
        # v >= 0
        # 1 - u - v >= 0
        return np.concatenate([u, v, 1.0 - u - v])
    
    cons = ({'type': 'ineq', 'fun': constraints})
    
    # Bounds for u, v: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Random init
        # u, v > 0, u+v < 1
        u = rng.random(N)
        v = rng.random(N)
        # Resample if invalid
        for _ in range(10):
            mask = (u + v < 1.0) & (u > 0) & (v > 0)
            if np.all(mask):
                break
            u[~mask] = rng.random(np.sum(~mask))
            v[~mask] = rng.random(np.sum(~mask))
        
        x0 = np.concatenate([u, v])
        
        try:
            res = minimize(objective_bary, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                params = res.x.reshape(N, 2)
                points = bary_to_cartesian(params)
                score = min_triangle_area_cartesian(points) / AREA_TRI
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
            
        if time.time() > deadline - 5.0:
            break
            
    if best_params is None:
        # Fallback
        return np.array([[0.5, 0.1], [0.1, 0.1], [0.9, 0.1], [0.5, 0.3], [0.3, 0.3], 
                         [0.7, 0.3], [0.5, 0.5], [0.2, 0.5], [0.8, 0.5], [0.5, 0.7], [0.5, 0.2]])
        
    return bary_to_cartesian(best_params)
