import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 100

def objective(params):
    # params: 100 values
    h = params
    # Ensure sum is 50? 
    # The problem states sum(h) = 50.
    # If we don't enforce it, we might violate constraints.
    # But DE can handle bounds.
    # We should enforce sum(h)=50 in the objective or parameterization.
    # Easier: parameterize 99 values, set 100th to 50 - sum(first 99).
    # But this might push h[99] out of [0,1].
    # Let's just use 100 variables and add a penalty for sum != 50.
    # Or, better, use a transformation that guarantees sum=50 and bounds [0,1].
    
    # Simple approach: 
    # h = params
    # Check bounds and sum.
    # If invalid, return large value.
    
    if np.any(h < 0) or np.any(h > 1):
        return 1e9
    if abs(np.sum(h) - 50) > 1e-6:
        return 1e9
        
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = 100.0 # 100 seconds
    
    # Initial guess: random or constant
    # Constant 0.5 is a valid seed.
    # Let's try to start with a random sequence that sums to 50.
    
    # DE bounds
    bounds = [(0, 1)] * N
    
    # DE is slow for 100 dimensions.
    # We might need to reduce dimensionality.
    # But the directive says "KEEP this design... investigate improvements within its family".
    # The family is "search".
    # Let's try DE with a small population and maxiter.
    
    # To speed up, we can use a lower dimension parameterization.
    # e.g., h_i = 0.5 + A * sin(2*pi*i*f + phi)
    # But this might be too restrictive.
    
    # Let's try a hybrid: 
    # 1. Random restarts with local optimization.
    
    best_val = np.max(np.correlate(np.full(N, 0.5), np.full(N, 0.5), mode="full"))
    best_h = np.full(N, 0.5)
    
    # Try a few random starts
    for i in range(10):
        if time.time() > deadline:
            break
            
        # Generate random h
        h_rand = np.random.rand(N)
        # Scale to sum 50
        # h_rand = h_rand / np.sum(h_rand) * 50
        # This might exceed 1.
        # Better: sort and clip?
        # Or just use DE on the raw values with penalty.
        
        res = differential_evolution(objective, bounds, maxiter=100, popsize=10, tol=1e-4, seed=i)
        
        if res.fun < best_val:
            best_val = res.fun
            best_h = res.x
            
    # Refine with L-BFGS-B
    # We need to handle the sum constraint.
    # Let's parameterize: x in R^99, h_100 = 50 - sum(x_1..x_99).
    # Bounds for x_i: [0,1].
    # Bound for h_100: [0,1] => sum(x) in [49, 50].
    
    def obj_refine(x):
        h = np.append(x, 50 - np.sum(x))
        if np.any(h < 0) or np.any(h > 1):
            return 1e9
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr)
    
    # Start refinement from best_h
    x0 = best_h[:99]
    
    res_refine = minimize(obj_refine, x0, method='L-BFGS-B', options={'maxiter': 1000})
    
    if res_refine.fun < best_val:
        best_val = res_refine.fun
        best_h = np.append(res_refine.x, 50 - np.sum(res_refine.x))
        
    # Ensure exact sum and bounds
    best_h = np.clip(best_h, 0, 1)
    # Adjust sum to be exactly 50
    diff = 50 - np.sum(best_h)
    best_h += diff / N
    
    return best_h
