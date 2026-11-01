import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin

    # Fix P0=(0,0) and P1=(1,0) to remove translation and rotation invariance.
    # This reduces variables to 2*(N-2) = 24.
    n_vars = 2 * (N - 2)
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        # Unpack x into the remaining N-2 points
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts

    def score(pts):
        """
        Calculate the score: min triangle area / convex hull area.
        Returns negative score for minimization.
        """
        try:
            # Check for degeneracy
            if np.any(np.diff(np.sort(pts[:, 0])) == 0) and np.any(np.diff(np.sort(pts[:, 1])) == 0):
                # Simple check, better to just calculate
                pass
            
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-9:
                return -1.0 # Invalid, minimize will avoid this if we return large positive for minimization? 
                           # Wait, we want to MAXIMIZE score. 
                           # So we minimize -score.
                           # If invalid, return a large positive number (bad score) for the minimizer.
                           # Actually, if score is 0, -score is 0. 
                           # Let's return a very large number for invalid to push it away?
                           # No, if hull area is 0, it's a degenerate line. 
                           # We want to avoid this.
                           # Let's just return -1e9 for invalid (which means score is -1e9, so -score is 1e9).
                           # This will be a huge penalty in minimization.
                return 1e9

            # Calculate min triangle area
            min_area = float('inf')
            # Vectorized calculation is possible but 364 is small
            # Use broadcasting for speed
            # i, j, k indices
            # This loop is fast enough
            for i in range(N):
                pi = pts[i]
                for j in range(i + 1, N):
                    p_ji = pts[j] - pi
                    for k in range(j + 1, N):
                        p_ki = pts[k] - pi
                        area = 0.5 * abs(p_ji[0] * p_ki[1] - p_ji[1] * p_ki[0])
                        if area < min_area:
                            min_area = area
            
            if min_area < 1e-9:
                return 1e9 # Degenerate triangle

            s = min_area / hull_area
            return -s # Minimize negative score

        except Exception:
            return 1e9

    def objective(x):
        pts = get_points(x)
        return score(pts)

    # Initial guess: Regular N-gon
    # Scale it so that distance between P0 and P1 is 1?
    # Regular polygon vertices on unit circle:
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # We need to map this to our parameterization (P0=0, P1=1)
    # This requires a transformation.
    # Easier: Just use a box of random points or the regular polygon coordinates directly 
    # and let the optimizer figure out the scale/position? 
    # No, we fixed P0 and P1. 
    # So we need to find a regular polygon where two vertices are (0,0) and (1,0).
    # This is a specific rotation and translation.
    # Let's just use a reasonable box for DE bounds.
    
    # Bounds for the 24 variables
    # Points should be within [-2, 2] to allow hull to form
    bounds = [(-2.0, 2.0)] * n_vars

    # Differential Evolution
    # maxiter and popsize tuned for speed
    # 14 points, 24 vars.
    # popsize=20, maxiter=100
    # tol=1e-5
    
    best_x = None
    best_val = float('inf')
    
    # Try DE
    try:
        # Use a seed for reproducibility in DE? No, we want diversity.
        # But construct() should be deterministic if possible? 
        # The task says "random.Random(N)" in seed, but we can use np.random if we want.
        # Let's use a fixed seed for DE to ensure consistent behavior if it works well.
        # Or just let it run.
        
        # We can provide an initial population? 
        # Let's just run DE.
        
        # To improve speed, we can limit the number of function evaluations.
        # maxiter=50, popsize=15 -> ~750 evals.
        # Each eval is ~0.1ms? 75ms.
        # DE overhead is higher.
        
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5,
            seed=42,
            polish=False # We will polish manually
        )
        
        if result.success or result.fun < best_val:
            best_x = result.x
            best_val = result.fun
            
    except Exception as e:
        pass

    # Local Polish with Powell
    if best_x is not None:
        # Check time
        if time.time() < deadline - 5.0:
            try:
                res_local = minimize(
                    objective, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res_local.fun < best_val:
                    best_x = res_local.x
                    best_val = res_local.fun
            except Exception:
                pass
    
    # If DE failed or didn't run, try a simple local search from a regular polygon guess
    if best_x is None:
        # Construct a regular polygon guess mapped to P0=0, P1=1
        # This is complex to do exactly. 
        # Let's just pick a random point in bounds and polish it.
        # Or use the regular polygon coordinates directly and optimize?
        # The parameterization is fixed. 
        # Let's just use a default initial guess: points on a circle of radius 1, centered at (0.5, 0.5)?
        # No, P0=0, P1=1.
        # Let's just use a random guess.
        np.random.seed(42)
        best_x = np.random.uniform(-1, 1, n_vars)
        
        if time.time() < deadline - 5.0:
            try:
                res_local = minimize(
                    objective, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                best_x = res_local.x
            except Exception:
                pass

    pts = get_points(best_x)
    return pts

# EVOLVE-BLOCK-END
