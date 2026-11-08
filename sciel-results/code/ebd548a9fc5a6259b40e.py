import numpy as np
import scipy.optimize as opt
import scipy.spatial as spatial
import time

N = 14

def construct():
    start_time = time.time()
    deadline = 100.0  # 100 seconds safety margin

    # Bounds for each coordinate: [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)

    def objective(params):
        # Check time limit
        if time.time() - start_time > deadline:
            # This is tricky to interrupt DE from inside.
            # Instead, we rely on the speed of the function.
            pass
        
        pts = params.reshape((N, 2))
        
        # Check for degenerate hull
        if np.std(pts, axis=0).max() < 1e-6:
            return 1e6

        try:
            hull = spatial.ConvexHull(pts)
            hull_area = hull.volume # In 2D, .volume is area
        except Exception:
            return 1e6

        if hull_area < 1e-6:
            return 1e6

        # Compute min triangle area
        min_tri_area = float('inf')
        # There are C(14,3) = 364 triangles
        # We can vectorize this or loop. Looping 364 times is fast.
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Area = 0.5 * |det(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_tri_area:
                        min_tri_area = area
        
        if min_tri_area == float('inf'):
            return 1e6

        score = min_tri_area / hull_area
        # We want to maximize score, so minimize -score
        return -score

    # Use Differential Evolution
    # popsize: 20
    # maxiter: 200
    # tol: 1e-8
    # seed: 42 for reproducibility
    
    # To ensure we don't timeout, we can run DE in a way that allows early stopping?
    # DE doesn't support easy early stopping based on time in scipy.
    # However, 28 dimensions is small.
    # Let's estimate cost.
    # 1 eval ~ 0.1 ms (very rough, likely faster).
    # 20 popsize * 200 iter * 28 dim (mutations) ~ 112,000 evaluations? No, DE is 1 population of 20, iterated 200 times.
    # So 4000 evaluations.
    # 4000 * 0.1 ms = 0.4 seconds.
    # We can safely increase maxiter to 1000.
    
    try:
        res = opt.differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1000, 
            tol=1e-10, 
            popsize=20,
            polish=True # Use L-BFGS-B to polish the result
        )
    except Exception:
        # Fallback to regular polygon if optimization fails
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        return pts

    best_params = res.x
    pts = best_params.reshape((N, 2))
    
    # Optional: Normalize the points? 
    # The score is invariant to scaling, so the raw coordinates don't matter for the score, 
    # but the return value should be the points. 
    # The task says "Place 14 points...". The evaluator normalizes by hull area.
    # So returning the points as found is fine.
    
    return pts
