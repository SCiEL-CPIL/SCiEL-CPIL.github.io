import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time

N = 13

def compute_min_triangle_area(points):
    """
    Vectorized computation of the minimum triangle area.
    points: (N, 2) array
    """
    n = points.shape[0]
    # Get all unique pairs of indices
    # We need triples (i, j, k) with i < j < k
    # Instead of loops, we can use broadcasting on indices
    # But for N=13, the number of triples is C(13,3) = 286.
    # A fully vectorized approach for all triples at once might use more memory but is faster.
    
    # Generate all combinations of 3 indices
    # This is small enough to just do it once or cache it? 
    # Since points change, we recompute.
    
    # Method: 
    # 1. Create indices i, j, k
    # 2. Extract points
    # 3. Compute areas
    
    # For N=13, explicit loops in Python are slow. 
    # Let's use a semi-vectorized approach or just optimize the loop.
    # Actually, for N=13, the loop overhead is significant.
    # Let's try to compute all areas using matrix operations.
    
    # p_i, p_j, p_k
    # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    
    # We can flatten the points to a list of coordinates? No, keep (N,2).
    
    # Let's stick to a slightly optimized loop for clarity and safety, 
    # but we can pre-compute the index combinations.
    
    # Pre-compute indices
    if not hasattr(compute_min_triangle_area, 'indices'):
        idxs = []
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    idxs.append((i, j, k))
        compute_min_triangle_area.indices = np.array(idxs)
        
    idxs = compute_min_triangle_area.indices
    # idxs shape: (286, 3)
    
    # Extract points: points[idxs[:, 0]], points[idxs[:, 1]], points[idxs[:, 2]]
    p1 = points[idxs[:, 0]] # (286, 2)
    p2 = points[idxs[:, 1]] # (286, 2)
    p3 = points[idxs[:, 2]] # (286, 2)
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def compute_hull_area(points):
    try:
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri_area = compute_min_triangle_area(points)
    return min_tri_area / hull_area

def construct():
    # Bounds for each coordinate
    # We normalize by hull area, so scale doesn't matter, but position does.
    # Using [-1, 1] is safe.
    bounds = [(-1, 1)] * (2 * N)
    
    def objective(x):
        points = x.reshape(N, 2)
        s = score(points)
        if s == 0.0:
            return 1e6
        return -s

    # Use Differential Evolution
    # popsize=10-15 is reasonable for 26 dims.
    # maxiter=100-200.
    # tol=1e-4
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for polish and overhead
    
    # We need a callback to stop if time is exceeded? 
    # differential_evolution doesn't have a built-in time stop, only maxiter.
    # We can estimate time per iteration. 
    # Alternatively, just set maxiter conservatively.
    # Let's try popsize=10, maxiter=100.
    
    # To be safe with the 120s limit, we run DE, then polish.
    
    # DE Strategy:
    # 'best1bin' or 'rand1bin'
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            popsize=10, 
            maxiter=100, 
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            init='latinhypercube'
        )
        
        best_x = res.x
        best_score = -res.fun
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit:
             return best_x.reshape(N, 2)
             
        # Polish with Nelder-Mead
        # Only if we have time
        remaining_time = time_limit - elapsed
        if remaining_time > 5:
            res2 = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.fun < best_score:
                best_x = res2.x
                best_score = -res2.fun
                
    except Exception as e:
        # Fallback to regular polygon if DE fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_x = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        
    return best_x.reshape(N, 2)
