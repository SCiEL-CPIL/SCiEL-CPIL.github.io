import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def construct():
    # We fix p0 at (0,0) and p1 at (1,0) to remove translation and rotation invariance.
    # This reduces the number of variables from 2*N to 2*(N-2) = 24.
    # The remaining points p2...p13 are free in R2.
    
    def get_points(x):
        # x is a vector of length 2*(N-2)
        # First 2 coords are p2, next 2 are p3, etc.
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts

    def score(pts):
        try:
            # Calculate areas of all triangles
            areas = []
            for i, j, k in itertools.combinations(range(N), 3):
                # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                det = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(det))
            min_area = min(areas)
            
            # Calculate convex hull area
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
            
            if hull_area < 1e-10:
                return -1.0  # Invalid configuration, return low score (high negative for minimization)
                
            return min_area / hull_area
        except Exception:
            return -1.0

    def objective(x):
        pts = get_points(x)
        s = score(pts)
        # We want to maximize s, so we minimize -s
        return -s

    # Bounds for the free points. 
    # Since the configuration is scale/rotation/translation invariant (modulo the fixed points),
    # the other points should be within a reasonable range.
    # If the hull is too large, the min area might be small relative to the hull.
    # A good bound is roughly [-5, 5] for each coordinate.
    bounds = [(-5.0, 5.0) for _ in range(2 * (N - 2))]
    
    # Use differential evolution for global search
    # maxiter and popsize control the trade-off between speed and quality.
    # 120s limit. N=14 is small, so this should be fast.
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15, 
        tol=1e-8,
        init='sobol'  # Sobol sequence for better coverage
    )
    
    best_x = result.x
    best_pts = get_points(best_x)
    
    # Refine with local optimization using a smooth approximation
    # We can use the log-sum-exp trick to smooth the min function.
    # However, the hull area is also non-smooth.
    # For a quick refinement, we can just use Powell or L-BFGS-B on the raw objective.
    # Since we are already at a local minimum of the non-smooth function, 
    # a local refinement might not improve much if the function is non-smooth.
    # But let's try a small local search to ensure we are at the bottom of the basin.
    
    # Actually, differential_evolution often gets stuck in local minima for high dimensions.
    # With 24 dimensions, it's borderline.
    # Let's add a local refinement step using 'Nelder-Mead' which is robust to non-smoothness.
    
    res_local = minimize(
        objective, 
        best_x, 
        method='Nelder-Mead', 
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    if res_local.fun < result.fun:
        best_x = res_local.x
        best_pts = get_points(best_x)
        
    return best_pts

# EVOLVE-BLOCK-END
