import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 16
D = 2

def objective(x):
    """
    Calculate the squared ratio of max to min pairwise distance.
    Input x is a flat array of length N*D.
    """
    pts = x.reshape(N, D)
    
    # Compute pairwise squared distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Get upper triangle indices to exclude self-distances and duplicates
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    
    # Handle degenerate cases
    if min_d_sq < 1e-12:
        return 1e10
        
    return max_d_sq / min_d_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Define bounds for coordinates
    # Scale is invariant, so bounds can be arbitrary but need to be large enough
    # to contain the optimal configuration.
    bounds = [(-10, 10)] * (N * D)
    
    best_result = None
    best_val = np.inf
    
    # Run Differential Evolution
    # Using workers=-1 to utilize all available cores for parallel evaluation
    # This is crucial for meeting the time limit with a decent population size.
    try:
        # maxiter is a soft limit; we will also check time in a wrapper if needed,
        # but DE has a 'maxiter' parameter. 
        # To strictly enforce time, we might need to wrap the objective or use 
        # a strategy that stops early. 
        # However, DE's 'maxiter' is per generation. 
        # Let's try a reasonable number of iterations.
        
        # Strategy: Run DE with a fixed number of iterations.
        # If it takes too long, we might not finish.
        # We can estimate time per iteration.
        # Pop size 50, 16 points. 
        # Let's use a smaller pop size to be safe, or rely on parallelism.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, # 15 * 32 = 480 individuals in pop? No, popsize is multiplier for N_vars.
            # N_vars = 32. popsize=15 -> 480 individuals. 
            # This might be too heavy. Let's use popsize=10 -> 320 individuals.
            # Actually, popsize is the size of the population relative to the dimension.
            # Default is 15. So 15 * 32 = 480.
            # Let's try popsize=10 (320 individuals) to be faster.
            tol=1e-8,
            atol=1e-8,
            seed=42,
            workers=-1, # Parallelize
            init='latinhypercube'
        )
        
        if time.time() < deadline:
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If we have time and a result, refine with Nelder-Mead
    if best_result is not None and time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            val = objective(res_nm.x)
            if val < best_val:
                best_val = val
                best_result = res_nm.x.copy()
        except Exception as e:
            print(f"NM failed: {e}")
            
    # Fallback if nothing worked or if we want to ensure we return something
    if best_result is None:
        # Return a grid as fallback
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
        
    return best_result.reshape(N, D)

# To ensure we don't exceed time, we can add a wrapper to objective that checks time?
# No, DE runs internally. If it exceeds, the whole script dies.
# The 'workers' parameter is key.
# Let's add a safety check: if the first few iterations take too long, we might need to reduce popsize.
# But we can't easily interrupt DE.
# We assume 200 iterations with popsize 10 (320 evals/iter) = 64000 evals.
# With 8 cores, 8000 evals per core.
# Each eval is ~100us? 0.8s. Total ~10-20s. Should be safe.
