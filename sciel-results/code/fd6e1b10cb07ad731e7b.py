import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 13

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    # Ensure points are float
    points = np.asarray(points, dtype=float)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    # Compute all triangle areas
    # Number of triangles is C(13, 3) = 286
    # We can vectorize this for speed
    # Indices for combinations
    i, j, k = np.indices((N, N, N))
    # Filter for i < j < k
    mask = (i < j) & (j < k)
    idx_i = i[mask]
    idx_j = j[mask]
    idx_k = k[mask]
    
    p_i = points[idx_i]
    p_j = points[idx_j]
    p_k = points[idx_k]
    
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # In 2D, cross product is a scalar: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Initial guess: Fermat spiral
    def initial_guess():
        n = N
        i = np.arange(n)
        r = np.sqrt(i / n)
        theta = i * 2.399963229728653  # Golden angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))
    
    # Differential Evolution
    # Bounds: We can use a reasonable box, e.g., [-5, 5]
    bounds = [(-5.0, 5.0)] * (2 * N)
    
    def objective(x):
        points = x.reshape(N, 2)
        return -_score(points)
    
    best_x = None
    best_score = -1.0
    
    # Run Differential Evolution
    try:
        # maxiter=50, popsize=10 is a good balance for 26 dims
        # We can increase if time allows
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=80, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=True  # Uses Nelder-Mead for final polish
        )
        
        if res.success or res.fun < best_score:
            best_x = res.x
            best_score = -res.fun
            
    except Exception as e:
        pass
        
    # If DE didn't run or failed, or if we have time, do a multi-start Nelder-Mead
    # Check time
    if time.time() < deadline:
        # Try a few local optimizations from different starts
        starts = [initial_guess()]
        # Add some random perturbations
        for _ in range(5):
            p = initial_guess()
            p += np.random.normal(0, 0.5, p.shape)
            starts.append(p)
            
        for start_p in starts:
            if time.time() > deadline - 5:
                break
            try:
                res_lm = minimize(
                    objective, 
                    start_p.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                score_lm = -res_lm.fun
                if score_lm > best_score:
                    best_score = score_lm
                    best_x = res_lm.x
            except Exception:
                pass

    # Fallback to initial guess if nothing worked
    if best_x is None:
        best_x = initial_guess().flatten()
        
    return best_x.reshape(N, 2)
