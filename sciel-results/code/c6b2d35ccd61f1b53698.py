import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Precompute triple indices for vectorized area calculation
    triples = list(itertools.combinations(range(N), 3))
    triples_arr = np.array(triples)
    p1_idx = triples_arr[:, 0]
    p2_idx = triples_arr[:, 1]
    p3_idx = triples_arr[:, 2]

    def objective(x):
        points = x.reshape(N, 2)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e6  # Return large positive value (bad score) for degenerate cases
        
        if hull_area < 1e-9:
            return 1e6
            
        # Calculate Triangle Areas
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        # Area = 0.5 * | (p2-p1) x (p3-p1) |
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        score = min_area / hull_area
        return -score

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_x = x0
    best_score = -objective(x0)

    # Try optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
        
        if time.time() < deadline:
            current_score = -res.fun
            if current_score > best_score:
                best_x = res.x
                best_score = current_score
    except Exception:
        pass

    # If time remains, try a few random restarts with perturbations
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution found so far
            noise = rng.normal(0, 0.1, size=(N, 2))
            x_start = best_x + noise
            
            try:
                res = minimize(objective, x_start, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-15})
                
                if time.time() < deadline:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_x = res.x
                        best_score = current_score
            except Exception:
                pass

    return best_x
