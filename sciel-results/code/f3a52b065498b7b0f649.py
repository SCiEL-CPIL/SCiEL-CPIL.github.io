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
        # Always try diverse starts to avoid getting stuck in local minima.
        # f14 (Different Powers) can have multiple local minima depending on the shift/rotation.
        
        if calls < budget - 20 and time_left():
            # Generate a set of diverse starting points
            # Mix of random and some structured points
            n_random = 20
            random_points = rng.uniform(-5, 5, size=(n_random, dim))
            
            # Add some deterministic points: center, and points along axes
            # This helps if the optimum is near the boundary or axes
            det_points = [np.zeros(dim)]
            for i in range(dim):
                det_points.append(np.array([-4.0 if j == i else 0.0 for j in range(dim)]))
                det_points.append(np.array([4.0 if j == i else 0.0 for j in range(dim)]))
            
            diverse_starts = np.vstack([random_points, np.array(det_points)])
            # Shuffle to ensure we don't just process randoms first if they are bad
            indices = rng.permutation(len(diverse_starts))
            diverse_starts = diverse_starts[indices]
            
            for x0 in diverse_starts:
                if calls >= budget - 10 or not time_left():
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
                    if best_f < 1e-6:
                        break
                except Exception:
                    break

        # Phase 3: Polishing
        # Use remaining budget to refine the best point.
        # L-BFGS-B is deterministic, so re-running from the same point with tighter
        # tolerances might yield slight improvements or confirm convergence.
        
        if calls < budget - 5 and time_left() and best_f < np.inf:
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

        # If we still have budget, try a few small perturbations around best_x
        # to ensure we haven't missed a slightly better nearby point due to numerical issues
        if calls < budget - 5 and time_left() and best_f < np.inf:
            n_perturbations = 5
            for i in range(n_perturbations):
                if calls >= budget - 2 or not time_left():
                    break
                
                # Small random perturbation
                scale = 0.05 / (i + 1)
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

        return best_x, best_f

# EVOLVE-BLOCK-END
