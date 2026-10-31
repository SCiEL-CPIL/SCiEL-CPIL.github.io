import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _get_points(x):
    """
    Map 22-dim vector x in [0,1]^22 to 11 points in the unit equilateral triangle.
    x is flattened: [u0, v0, u1, v1, ..., u10, v10]
    """
    x = np.reshape(x, (N, 2))
    u = x[:, 0]
    v = x[:, 1] * (1 - u)
    w = 1 - u - v
    
    # Barycentric to Cartesian
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # P = u*A + v*B + w*C
    px = 0.5 * w + 1.0 * v
    py = (np.sqrt(3) / 2) * w
    
    return np.column_stack((px, py))

def _min_triangle_area(points):
    """
    Calculate the minimum area of triangles formed by the points.
    points: (N, 2) array
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function for minimization.
    Returns negative of the minimum triangle area.
    """
    points = _get_points(x)
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    best_x = None
    best_val = -np.inf
    
    # Strategy: Multi-start local optimization
    # We will generate random initial points and optimize them locally.
    # Powell method is good for non-smooth functions.
    
    np.random.seed(42)
    
    # Generate a batch of initial points
    # We can also add some structured points like vertices and midpoints to help
    # But random is usually sufficient with enough restarts.
    
    num_restarts = 200  # We will run as many as time allows
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Generate random initial point in barycentric coords
        # u, v, w > 0, u+v+w=1
        # Simple way: generate 22 random numbers, map to u, v
        # To ensure points are distinct and spread out, we can use a simple random uniform in the triangle
        x0 = np.random.rand(2 * N)
        
        # Try to optimize
        try:
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 10000}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        # Optional: If time permits, do a second polish with Nelder-Mead on the best found so far
        # This can sometimes refine the solution further.
        if i % 10 == 0 and best_x is not None:
            try:
                res_nm = minimize(
                    objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'maxfev': 5000}
                )
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

    if best_x is None:
        # Fallback
        best_x = np.random.rand(2 * N)
        best_val = objective(best_x)
        
    points = _get_points(best_x)
    return points
