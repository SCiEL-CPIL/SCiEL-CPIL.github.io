# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses symmetry-based global optimization (D3h and D7h symmetries) followed by local refinement.
    """
    
    start_time = time.time()
    time_limit = 600 # Safety margin
    
    def get_points_D3h(params):
        a, b, phi = params
        r = 1.0
        points = np.zeros((14, 3))
        # Pole 1
        points[0] = [0, 0, a]
        # Pole 2
        points[1] = [0, 0, -a]
        # Ring 1 (6 points)
        for i in range(6):
            theta = i * np.pi / 3
            points[2 + i] = [r * np.cos(theta), r * np.sin(theta), b]
        # Ring 2 (6 points)
        for i in range(6):
            theta = i * np.pi / 3 + phi
            points[8 + i] = [r * np.cos(theta), r * np.sin(theta), -b]
        return points

    def get_points_2rings7(params):
        b, phi = params
        r = 1.0
        points = np.zeros((14, 3))
        # Ring 1 (7 points)
        for i in range(7):
            theta = i * 2 * np.pi / 7
            points[i] = [r * np.cos(theta), r * np.sin(theta), b]
        # Ring 2 (7 points)
        for i in range(7):
            theta = i * 2 * np.pi / 7 + phi
            points[7 + i] = [r * np.cos(theta), r * np.sin(theta), -b]
        return points

    def metric(points):
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_D3h(params):
        points = get_points_D3h(params)
        return -metric(points)

    def objective_2rings7(params):
        points = get_points_2rings7(params)
        return -metric(points)

    best_points = None
    best_score = -1.0

    # 1. Optimize D3h symmetry (2 poles + 2 hex rings)
    # Bounds: a, b in [0, 2], phi in [0, pi/6]
    bounds_D3h = [(0, 2), (0, 2), (0, np.pi / 6)]
    try:
        res_D3h = differential_evolution(
            objective_D3h, 
            bounds_D3h, 
            seed=42, 
            maxiter=1000, 
            popsize=20,
            tol=1e-8
        )
        if not res_D3h.success:
             # Fallback to best individual if success flag is false but we have a result
             pass 
        best_params_D3h = res_D3h.x
        points_D3h = get_points_D3h(best_params_D3h)
        score_D3h = metric(points_D3h)
        
        if score_D3h > best_score:
            best_score = score_D3h
            best_points = points_D3h
            
    except Exception as e:
        pass

    # Check time
    if time.time() - start_time > time_limit * 0.5:
        return best_points if best_points is not None else np.random.randn(14, 3)

    # 2. Optimize 2 rings of 7 symmetry
    bounds_2rings7 = [(0, 2), (0, 2 * np.pi / 7)]
    try:
        res_2rings7 = differential_evolution(
            objective_2rings7, 
            bounds_2rings7, 
            seed=42, 
            maxiter=1000, 
            popsize=20,
            tol=1e-8
        )
        best_params_2rings7 = res_2rings7.x
        points_2rings7 = get_points_2rings7(best_params_2rings7)
        score_2rings7 = metric(points_2rings7)
        
        if score_2rings7 > best_score:
            best_score = score_2rings7
            best_points = points_2rings7
            
    except Exception as e:
        pass

    # 3. Local refinement of the best symmetric solution
    # Break symmetry by optimizing all 42 coordinates
    if best_points is not None and time.time() - start_time < time_limit * 0.8:
        try:
            def objective_full(points_flat):
                points = points_flat.reshape(14, 3)
                return -metric(points)
            
            # Use Nelder-Mead or L-BFGS-B. Nelder-Mead is robust for non-smooth min/max.
            # L-BFGS-B is faster but might get stuck in local minima of the min-distance function.
            # Let's use a few restarts with Nelder-Mead.
            
            # Initial guess is the best symmetric point
            x0 = best_points.flatten()
            
            # Run a few local optimizations
            for i in range(10):
                # Add small noise to escape local minima
                if i > 0:
                    noise = np.random.normal(0, 0.01, size=x0.shape)
                    x_start = x0 + noise
                else:
                    x_start = x0
                
                res_local = minimize(
                    objective_full, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000}
                )
                
                points_local = res_local.x.reshape(14, 3)
                score_local = metric(points_local)
                
                if score_local > best_score:
                    best_score = score_local
                    best_points = points_local
                    x0 = res_local.x # Update best start point

        except Exception as e:
            pass

    # Final sanity check
    if best_points is None:
        # Fallback to random if everything failed
        np.random.seed(42)
        best_points = np.random.randn(14, 3)

    return best_points

# EVOLVE-BLOCK-END
