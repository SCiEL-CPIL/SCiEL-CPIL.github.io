"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a prioritized Multi-Start L-BFGS-B strategy.
    
    1. Start with the origin, which is often close to the BBOB optimum.
    2. Run L-BFGS-B from the origin.
    3. If the result is not excellent, explore diverse points (Halton, Random).
    4. Use remaining budget to polish the best solution found.
    
    L-BFGS-B is highly efficient for smooth functions, allowing rapid convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0
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
            return deadline - time.time() > 1.0

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate Halton sequence points
        def halton_sequence(n, dims):
            points = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
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

        rng = np.random.default_rng(12345)
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Phase 1: Origin Start
        # Origin is a strong prior for BBOB instances
        x0_origin = np.zeros(dim)
        try:
            f_origin = wrapped_func(x0_origin)
            update_best(x0_origin, f_origin)
            
            # Run L-BFGS-B from origin
            res = minimize(
                wrapped_func, 
                x0_origin, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            update_best(res.x, res.fun)
        except Exception:
            pass

        # Phase 2: Diverse Exploration
        # If the origin start didn't find a very good solution, try others.
        # Threshold: if best_f > 0.1, we might be in a bad basin or far from optimum.
        if best_f > 0.1 and calls < budget - 50 and time_left():
            # Generate Halton points
            n_halton = 5
            halton_points = halton_sequence(n_halton, dim)
            halton_mapped = map_to_box(halton_points)
            
            # Generate Random points
            n_random = 5
            random_points = rng.uniform(-5, 5, size=(n_random, dim))
            
            # Combine and shuffle to avoid bias
            diverse_starts = np.vstack([halton_mapped, random_points])
            indices = rng.permutation(len(diverse_starts))
            diverse_starts = diverse_starts[indices]
            
            for x0 in diverse_starts:
                if calls >= budget or not time_left():
                    break
                
                try:
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 50, 'gtol': 1e-6}
                    )
                    update_best(res.x, res.fun)
                    
                    # If we found a very good solution, stop exploring
                    if best_f < 1e-4:
                        break
                except Exception:
                    break

        # Phase 3: Polishing
        # Use remaining budget to refine the best point.
        # L-BFGS-B is deterministic, so re-running from the same point with tighter
        # tolerances might yield slight improvements or confirm convergence.
        # We also try small perturbations to ensure we are at the true minimum.
        
        if calls < budget - 10 and time_left() and best_f < np.inf:
            # Tight polish on best_x
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-10}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

        # If we still have budget, try a few small perturbations
        if calls < budget - 20 and time_left() and best_f < np.inf:
            n_perturbations = 5
            for i in range(n_perturbations):
                if calls >= budget or not time_left():
                    break
                
                # Small random perturbation
                scale = 0.1 / (i + 1)
                perturbation = rng.normal(0, scale, size=dim)
                x_perturbed = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    res = minimize(
                        wrapped_func, 
                        x_perturbed, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 30, 'gtol': 1e-8}
                    )
                    update_best(res.x, res.fun)
                except Exception:
                    break

        # Final check: if we have lots of budget left, run a few more diverse starts
        # This is a safety net in case the initial origin start was bad and the
        # threshold in Phase 2 was too strict.
        if calls < budget - 50 and time_left():
             # Generate more random points
             n_more = 5
             more_random = rng.uniform(-5, 5, size=(n_more, dim))
             for x0 in more_random:
                 if calls >= budget or not time_left():
                     break
                 try:
                     res = minimize(
                         wrapped_func, 
                         x0, 
                         method='L-BFGS-B', 
                         bounds=bounds,
                         options={'maxiter': 50, 'gtol': 1e-6}
                     )
                     update_best(res.x, res.fun)
                 except Exception:
                     break

        return best_x, best_f

# EVOLVE-BLOCK-END
