import numpy as np
from scipy.optimize import minimize
import time

N = 11
V = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2]
])

# Precompute triangle indices
indices = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            indices.append((i, j, k))
indices = np.array(indices)

def compute_min_area(points):
    # points shape (N, 2)
    # p_i, p_j, p_k
    pi = points[indices[:, 0]]
    pj = points[indices[:, 1]]
    pk = points[indices[:, 2]]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Using cross product magnitude
    # v1 = pj - pi, v2 = pk - pi
    # Area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def objective(x):
    points = x.reshape(N, 2)
    # Check if points are inside? SLSQP handles constraints, but if we start outside, 
    # the objective value might be weird if we don't penalize. 
    # However, SLSQP projects back or handles it. 
    # It's safer to just compute area.
    return -compute_min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Constraints
    # y >= 0
    # y <= sqrt(3) x
    # y <= sqrt(3) (1-x)
    
    # Inequality constraints: func(x) >= 0
    # 1. y_i >= 0
    # 2. sqrt(3) x_i - y_i >= 0
    # 3. sqrt(3) (1-x_i) - y_i >= 0
    
    def con_y_pos(x):
        points = x.reshape(N, 2)
        return points[:, 1]
        
    def con_left(x):
        points = x.reshape(N, 2)
        return np.sqrt(3) * points[:, 0] - points[:, 1]
        
    def con_right(x):
        points = x.reshape(N, 2)
        return np.sqrt(3) * (1 - points[:, 0]) - points[:, 1]

    # We need to combine these into a single constraint function or list
    # SLSQP accepts a list of constraint dicts or a function returning array
    # If using function, it should return array of values >= 0
    
    def constraints(x):
        points = x.reshape(N, 2)
        c1 = points[:, 1]
        c2 = np.sqrt(3) * points[:, 0] - points[:, 1]
        c3 = np.sqrt(3) * (1 - points[:, 0]) - points[:, 1]
        return np.concatenate([c1, c2, c3])

    best_val = -np.inf
    best_points = None
    
    # Initial guesses
    # 1. Center of gravity?
    # 2. Random points
    
    rng = np.random.default_rng(42)
    
    # Try a few random restarts
    max_restarts = 50
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Generate random points inside triangle
        # Use barycentric coordinates
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        # Barycentric to Cartesian
        # P = u*V0 + v*V1 + (1-u-v)*V2
        # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
        # x = u*0 + v*1 + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
        # y = u*0 + v*0 + (1-u-v)*sqrt(3)/2 = (1-u-v)*sqrt(3)/2
        
        x_coords = 0.5 + 0.5*v - 0.5*u
        y_coords = (1 - u - v) * (np.sqrt(3) / 2)
        
        x0 = np.column_stack((x_coords, y_coords)).flatten()
        
        # Add some noise or specific patterns for diversity?
        # For first few, maybe use structured grids?
        # Let's stick to random for now, SLSQP is good at escaping local minima if started well.
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=constraints,
                bounds=[(0, 1)] * (2 * N), # x in [0,1], y in [0, sqrt(3)/2] approx
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_points = res.x.reshape(N, 2)
        except Exception as e:
            pass

    if best_points is None:
        # Fallback: return random points
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x_coords = 0.5 + 0.5*v - 0.5*u
        y_coords = (1 - u - v) * (np.sqrt(3) / 2)
        best_points = np.column_stack((x_coords, y_coords))

    return best_points
