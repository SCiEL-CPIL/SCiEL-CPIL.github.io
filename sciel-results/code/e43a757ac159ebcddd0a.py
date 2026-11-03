import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def bary_to_cart(uv):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    w = 1 - u - v
    P = u*V0 + v*V1 + w*V2
    """
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack((x, y))

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    min_a = float('inf')
    n = len(points)
    # Precompute differences to speed up? No, N is small.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            v1 = points[j] - pi
            for k in range(j + 1, n):
                v2 = points[k] - pi
                # Cross product magnitude in 2D
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_a:
                    min_a = area
    return min_a

def objective(uv_flat):
    """
    Objective function for minimization.
    We minimize -min_area + penalty.
    """
    uv = uv_flat.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Constraint: u >= 0, v >= 0, u + v <= 1
    # Penalty for violations
    viol = 0.0
    # u < 0
    neg_u = np.clip(-u, 0, None)
    viol += np.sum(neg_u * neg_u)
    # v < 0
    neg_v = np.clip(-v, 0, None)
    viol += np.sum(neg_v * neg_v)
    # u + v > 1
    over = np.clip(u + v - 1, 0, None)
    viol += np.sum(over * over)
    
    if viol > 1e-8:
        return 1000.0 * viol
        
    points = bary_to_cart(uv)
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    best_score = -float('inf')
    best_uv = None
    
    rng = np.random.default_rng(7)
    
    # Generate initial guesses
    guesses = []
    
    # 1. Random guesses
    for _ in range(10):
        u = rng.random(N)
        # v must be in [0, 1-u]
        v = rng.random(N) * (1.0 - u)
        guesses.append(np.column_stack((u, v)).flatten())
        
    # 2. Structured guess: Grid
    # Try to place points in a 3x3 grid + 2 extra
    # Just use a simple grid for one guess
    grid_u = []
    grid_v = []
    # 3x3 grid in barycentric
    # u in {0, 0.5, 1}, v in {0, 0.5, 1} but must satisfy u+v<=1
    # Valid: (0,0), (0,0.5), (0,1), (0.5,0), (0.5,0.5), (1,0) -> 6 points
    # We need 11.
    # Let's just rely on random restarts, they are robust.
    
    for guess in guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            # Check if the result is valid (low penalty)
            val = res.fun
            if val < 1.0: # Penalty is 1000*viol^2, so if val < 1, viol is small
                score = -val
                if score > best_score:
                    best_score = score
                    best_uv = res.x
        except Exception:
            pass
            
    if best_uv is None:
        # Fallback: Return random points if optimization failed
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_uv = np.column_stack((u, v)).flatten()
        
    points = bary_to_cart(best_uv.reshape(N, 2))
    return points
