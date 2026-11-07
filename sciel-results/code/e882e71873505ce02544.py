"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using Multi-Start L-BFGS-B for BBOB f14.
    
    Strategy:
    1. Generate diverse starting points using Halton sequence + random.
    2. Apply L-BFGS-B from each start - it converges rapidly for smooth functions.
    3. Early progress: L-BFGS-B achieves low error in few iterations, boosting AOCC.
    4. Strict budget and time management with safety margins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 2.0

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate starting points
        def halton_sequence(n, dims):
            points = []
            primes = [2, 3, 5, 7, 11]
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = primes[d]
                    result = 0.0
                    denom = 1
                    ii = i
                    while True:
                        denom *= f
                        result += (ii % f) / denom
                        ii //= f
                        if ii == 0:
                            break
                    point.append(result)
                points.append(point)
            return np.array(points)

        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Generate diverse starts: Halton + origin + random
        n_halton = 6
        halton_points = halton_sequence(n_halton, dim)
        start_points = map_to_box(halton_points)
        
        # Add origin and a few random points
        rng = np.random.default_rng(12345)
        random_points = rng.uniform(-5, 5, size=(3, dim))
        start_points = np.vstack([start_points, np.zeros(dim), random_points])
        
        # Shuffle for diversity
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim
        top_candidates = []

        # Run L-BFGS-B from each start
        # L-BFGS-B is efficient: ~50-100 calls per run for good convergence
        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 150, 'gtol': 1e-12, 'ftol': 1e-15, 'maxfun': 100}
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                
                top_candidates.append((res.fun, res.x.copy()))
                
            except (ValueError, Exception) as e:
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    break
                else:
                    # Continue with next start
                    pass

        # If budget remains, do a final refinement from the best candidate
        if top_candidates and calls < budget and time_left():
            top_candidates.sort(key=lambda item: item[0])
            best_cand_f, best_cand_x = top_candidates[0]
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_cand_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-14, 'ftol': 1e-18, 'maxfun': 80}
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except (ValueError, Exception) as e:
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    pass
                else:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
