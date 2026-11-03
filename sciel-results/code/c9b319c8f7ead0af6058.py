"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """Objective function for optimization: negative of minimum area."""
    points = x.reshape(N, 2)
    # Check if points are inside the triangle
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Constraints:
    # y >= 0
    # y <= sqrt(3) * x  (line from (0,0) to (0.5, sqrt(3)/2))
    # y <= -sqrt(3) * x + sqrt(3) (line from (1,0) to (0.5, sqrt(3)/2))
    
    penalty = 0.0
    for i in range(N):
        x_i, y_i = points[i]
        if y_i < 0:
            penalty += (0 - y_i) ** 2
        if y_i > np.sqrt(3) * x_i:
            penalty += (y_i - np.sqrt(3) * x_i) ** 2
        if y_i > -np.sqrt(3) * x_i + np.sqrt(3):
            penalty += (y_i - (-np.sqrt(3) * x_i + np.sqrt(3))) ** 2
        
    area = min_triangle_area(points)
    return -(area / TRI_AREA) + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Bounds for differential evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
    
    best_x = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search
    # This is robust for non-smooth, non-convex problems
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            update_interval=1,
            polish=True, # Use L-BFGS-B to polish the best result
            workers=-1    # Use all cores if possible
        )
        
        if result.success or result.fun < 0: # Check if we found a valid solution
            x_opt = result.x.reshape(N, 2)
            # Verify validity strictly
            valid = True
            for i in range(N):
                x_i, y_i = x_opt[i]
                if y_i < -1e-9 or y_i > np.sqrt(3) * x_i + 1e-9 or y_i > -np.sqrt(3) * x_i + np.sqrt(3) + 1e-9:
                    valid = False
                    break
            
            if valid:
                score = min_triangle_area(x_opt) / TRI_AREA
                if score > best_score:
                    best_score = score
                    best_x = x_opt
    except Exception as e:
        pass

    # If DE failed or didn't find a good solution, try a few local refinements from random starts
    if time.time() < deadline:
        for seed in [1, 2, 3]:
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            # Generate random points inside triangle
            x0 = np.zeros((N, 2))
            for i in range(N):
                u, v = rng.random(), rng.random()
                while u + v > 1.0:
                    u, v = rng.random(), rng.random()
                x0[i, 0] = u + 0.5 * v
                x0[i, 1] = np.sqrt(3) / 2 * v
            
            try:
                res = minimize(
                    objective,
                    x0.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0,1),(0,np.sqrt(3)/2)]*N,
                    options={'maxiter': 200, 'ftol': 1e-6}
                )
                x_opt = res.x.reshape(N, 2)
                valid = True
                for i in range(N):
                    x_i, y_i = x_opt[i]
                    if y_i < -1e-9 or y_i > np.sqrt(3) * x_i + 1e-9 or y_i > -np.sqrt(3) * x_i + np.sqrt(3) + 1e-9:
                        valid = False
                        break
                if valid:
                    score = min_triangle_area(x_opt) / TRI_AREA
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
            except Exception:
                pass

    if best_x is None:
        # Fallback: return a simple valid configuration
        # e.g., vertices and center
        best_x = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/6], # centroid
            [0.25, np.sqrt(3)/12], [0.75, np.sqrt(3)/12], [0.5, np.sqrt(3)/4],
            [0.125, np.sqrt(3)/24], [0.875, np.sqrt(3)/24], [0.375, np.sqrt(3)/8], [0.625, np.sqrt(3)/8]
        ])
        # Ensure they are inside
        # The above points are inside.
        
    return best_x
# EVOLVE-BLOCK-END
