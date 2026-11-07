import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        # Use a fixed seed for reproducibility within the instance
        # The instance itself changes the function, so this is fine.
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            # Clip to bounds to be safe, although the problem says func clips.
            # But we should probably not double clip if func does it?
            # "func(x) clips a finite length-5 vector to the box"
            # So we can just pass x.
            val = func(x)
            calls_made += 1
            return val
        
        # Phase 1: Random Sampling (Exploration)
        # Katsuura is multimodal. We need a good spread.
        # Budget allocation: ~20% for random sampling, 80% for local refinement.
        n_random = max(10, int(0.2 * budget))
        
        # Generate random points in [-5, 5]
        # Uniform distribution
        points = rng.uniform(-5, 5, size=(n_random, dim))
        
        for x in points:
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
        
        # Phase 2: Local Refinement (Exploitation)
        # Use Nelder-Mead on the best points found in Phase 1.
        # Sort the random points by fitness to pick the best candidates for refinement.
        # We only have the best_f and best_x so far, but we can re-evaluate or store.
        # Better: store all evaluated points and their fitness.
        
        # Let's restructure to store points.
        
        # Reset state for a cleaner implementation
        best_x = None
        best_f = np.inf
        calls_made = 0
        evaluated_points = []
        
        def safe_call_stored(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            val = func(x)
            calls_made += 1
            evaluated_points.append((x, val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Random Sampling
        n_random = max(10, int(0.2 * budget))
        points = rng.uniform(-5, 5, size=(n_random, dim))
        
        for x in points:
            safe_call_stored(x)
            if calls_made >= budget or time.time() > deadline:
                break
        
        # Sort evaluated points by fitness
        evaluated_points.sort(key=lambda item: item[1])
        
        # Phase 2: Nelder-Mead on top candidates
        # How many candidates?
        # We have budget - calls_made left.
        # Each Nelder-Mead run in dim=5 takes roughly 2*dim to 10*dim evaluations depending on convergence.
        # Let's say ~50-100 evals per run.
        # We can afford (budget - calls_made) / 50 runs.
        # But we should prioritize the best ones.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f
        
        # Number of local optimizations to run
        # Let's try to run on the top K points.
        # K should be such that K * avg_cost <= remaining_budget
        avg_cost = 50  # rough estimate for Nelder-Mead in 5D
        num_local = min(len(evaluated_points), max(1, int(remaining_budget / avg_cost)))
        
        # Select top num_local unique points (avoiding duplicates if any, though unlikely with continuous random)
        # We just take the first num_local from the sorted list.
        
        for i in range(num_local):
            if calls_made >= budget or time.time() > deadline:
                break
            
            x0 = evaluated_points[i][0]
            
            # Nelder-Mead parameters
            # maxiter is important to prevent hanging
            # maxfev is also good to cap
            maxfev = min(100, budget - calls_made)
            if maxfev < 10:
                break
                
            # Use a callback to check time/budget?
            # scipy.optimize.minimize doesn't have a built-in time check,
            # but maxfev limits calls.
            # We can wrap the function to check time.
            
            def wrapped_func(x):
                if time.time() > deadline or calls_made >= budget:
                    return best_f # or np.inf? best_f is safer to stop early if it's worse
                return safe_call_stored(x)
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'maxfev': maxfev, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                # The result x is res.x, but we already updated best_x in safe_call_stored
                # Check if res.fun is better? No, safe_call_stored updates best_f.
                
            except Exception:
                pass
        
        return best_x, best_f
