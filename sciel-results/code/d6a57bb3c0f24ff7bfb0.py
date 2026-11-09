import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 8
    # Differential evolution operates on a flat vector.
    # We parameterize points as (r_i, theta_i) for i=0..7.
    # r_i in [0.1, 1.5], theta_i in [0, 2*pi).
    # This parameterization avoids singularities at the origin (r=0) 
    # and keeps points within a reasonable bounded box for the optimizer,
    # while the objective function handles the scaling.
    
    def compute_log_score_vectorized(points):
        """
        points: (n, 2) array
        Returns log(Delta / n^n)
        """
        # Compute pairwise squared distances
        # diff[i, j, k] = points[i, k] - points[j, k]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        valid_dists_sq = dists_sq[mask]
        
        # Check for distinct points
        if np.any(valid_dists_sq < 1e-10):
            return -np.inf
            
        D_sq = np.max(valid_dists_sq)
        D = np.sqrt(D_sq)
        
        if D < 1e-10:
            return -np.inf
            
        # log(Delta) = sum_{i!=j} log(d_ij * 2/D)
        #            = sum(log(d_ij)) + n(n-1) * (log(2) - log(D))
        # sum(log(d_ij)) = 0.5 * sum(log(d_ij^2))
        
        log_prod_unscaled = 0.5 * np.sum(np.log(valid_dists_sq))
        log_scaled_prod = log_prod_unscaled + n * (n - 1) * (np.log(2.0) - np.log(D))
        
        # log(Score) = log_scaled_prod - n * log(n)
        return log_scaled_prod - n * np.log(n)

    def objective(x):
        """
        Objective function for differential_evolution (minimization).
        x is a flat array of 16 parameters: [r0, t0, r1, t1, ..., r7, t7]
        """
        # Convert polar to cartesian
        r = x[0::2]
        t = x[1::2]
        
        # Ensure r is positive (bounds handle this, but just in case)
        if np.any(r < 1e-6):
            return 1e10
            
        points = np.column_stack([r * np.cos(t), r * np.sin(t)])
        
        # Optional: Check for collisions in cartesian space
        # If two points are very close, the score is -inf
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(n, dtype=bool)
        valid_dists_sq = dists_sq[mask]
        
        if np.any(valid_dists_sq < 1e-8):
            return 1e10

        s = compute_log_score_vectorized(points)
        if not np.isfinite(s):
            return 1e10
            
        return -s

    # Bounds for Differential Evolution
    # r: [0.1, 1.5] (Points shouldn't be too close to origin or too far, 
    #    though scaling makes absolute distance irrelevant, this helps DE converge)
    # theta: [0, 2*pi]
    bounds = [(0.1, 1.5), (0, 2*np.pi)] * n
    
    # Initial population strategy: 
    # DE generates its own population, but we can guide it with a good seed if we use 'init'
    # However, standard DE is robust. Let's run it.
    
    # We will run Differential Evolution.
    # maxiter: number of generations. 
    # popsize: size of the population.
    # tol: tolerance for stopping.
    
    # Time budget: ~145s.
    # DE can be slow if popsize is large. 
    # n=8, dim=16. 
    # Let's try popsize=20, maxiter=200.
    # Each iteration evaluates popsize * dim * some_factor points.
    # 20 * 16 * 50 = 16000 evaluations per generation? No, DE is more efficient.
    # Let's estimate: 1 eval ~ 1ms. 145s / 1ms = 145,000 evals.
    # popsize=20, maxiter=100 -> 2000 evals per generation? No, DE creates new points.
    # Actually, DE evaluates the new points. 
    # Let's set a time-based stopper if possible, or just conservative limits.
    
    # scipy differential_evolution has a 'maxiter' but no direct time limit.
    # We can wrap the objective to check time, but that's messy.
    # Instead, we'll run a few moderate DE runs if time permits, or one long one.
    
    # Let's run one DE run with reasonable parameters.
    # If it finishes quickly, we can do another with different settings.
    
    best_x = None
    best_val = 1e10
    
    # Strategy: 
    # 1. Run DE with popsize=20, maxiter=100
    # 2. If time remains, run DE with popsize=30, maxiter=50 (different exploration)
    # 3. Fine tune best result with L-BFGS-B
    
    try:
        # Run 1
        res1 = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            polish=True,  # Use L-BFGS-B to polish the result at the end
            init='sobol' # Better initial spread
        )
        
        if res1.success or res1.fun < best_val:
            best_val = res1.fun
            best_x = res1.x
            
    except Exception:
        pass

    # Check time
    if time.time() < deadline - 20:
        try:
            # Run 2: Different population size and seed
            res2 = differential_evolution(
                objective, 
                bounds, 
                maxiter=80, 
                popsize=30, 
                tol=1e-10, 
                seed=123,
                polish=True,
                init='latinhypercube'
            )
            
            if res2.success or res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
        except Exception:
            pass

    # Final Polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            res_polish = minimize(
                objective, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_polish.success or res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x
        except Exception:
            pass

    # If DE failed or didn't find anything better than 1.0 (octagon), 
    # we should return the octagon.
    # Score 1.0 corresponds to log_score = 0.
    # So objective value = 0.
    
    # Let's evaluate the regular octagon to have a fallback
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Convert octagon to polar parameters for objective function
    r_oct = np.ones(n)
    t_oct = angles
    x_oct = np.empty(2*n)
    x_oct[0::2] = r_oct
    x_oct[1::2] = t_oct
    
    val_oct = objective(x_oct)
    
    if best_x is None or best_val > val_oct:
        best_x = x_oct
        best_val = val_oct

    # Convert best_x (polar) back to cartesian
    r = best_x[0::2]
    t = best_x[1::2]
    points = np.column_stack([r * np.cos(t), r * np.sin(t)])
    
    # Scale points so that diameter is 2? 
    # The task says: "The score is Delta/n^n, where Delta is the product ... after scaling the set to diameter 2"
    # The evaluator scales the points itself. 
    # "construct() -> 8 distinct points [x, y] (floats, |coordinate| < 10^6)"
    # So we just need to return valid points. The evaluator handles the scaling for scoring.
    # However, returning points with diameter 2 is cleaner and avoids any potential floating point issues 
    # if the evaluator doesn't scale correctly (though the contract says it does).
    # Let's scale to diameter 2 just to be safe and consistent with the "normalized" view.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    D = np.max(valid_dists)
    
    if D > 1e-10:
        points = points * (2.0 / D)
    
    # Check distinctness one last time
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists < 1e-5):
        # Fallback to regular octagon
        points = pts_oct
        # Scale octagon to diameter 2
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        D = np.max(dists[~np.eye(n, dtype=bool)])
        if D > 0:
            points = points * (2.0 / D)

    return [[float(x), float(y)] for x, y in points]

# Test locally
if __name__ == "__main__":
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Points: {pts}")
    
    # Calculate score
    n = 8
    diff = np.array(pts)[:, np.newaxis, :] - np.array(pts)[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    D = np.max(valid_dists)
    scaled_dists = valid_dists * (2.0 / D)
    Delta = np.prod(scaled_dists)
    score = Delta / (n ** n)
    print(f"Score: {score}")
