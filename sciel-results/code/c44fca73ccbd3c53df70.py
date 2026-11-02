import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

N = 13

def construct():
    def objective(params):
        pts = params.reshape(N, 2)
        
        # Check for degenerate cases (all points same, etc.)
        if np.allclose(pts, pts[0]):
            return 1.0
            
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            return 1.0
            
        if hull_area < 1e-10:
            return 1.0

        # Calculate min triangle area
        min_area = np.inf
        # N=13, 286 triples
        # Vectorization trick:
        # We can compute all pairwise differences?
        # Let's just loop for clarity and speed is likely fine.
        for i in range(N):
            pi = pts[i]
            for j in range(i+1, N):
                v1 = pts[j] - pi
                for k in range(j+1, N):
                    v2 = pts[k] - pi
                    area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        
        score = min_area / hull_area
        return -score

    bounds = [(-1, 1)] * (2 * N)
    
    # Differential Evolution
    # Use a moderate population and iterations
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=100, 
        tol=1e-5, 
        polish=True, 
        popsize=15,
        init='sobol' # Sobol sequence is good for high dimensions
    )
    
    best_pts = res.x.reshape(N, 2)
    
    return best_pts
