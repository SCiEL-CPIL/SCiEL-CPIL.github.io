import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f9 (rotated Rosenbrock).
    Combines Differential Evolution for global exploration with multi-start L-BFGS-B for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check
            if time.time() > deadline:
                return best_f
            # Budget check
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Search using Differential Evolution
        # DE is robust to multimodality and good for finding the basin of attraction.
        # We allocate ~50% of the budget to this phase to leave more for polishing.
        de_budget = int(budget * 0.5)
        
        # DE parameters:
        # popsize: 12 (slightly smaller to save budget for more iterations or polishing)
        # maxiter: estimated to fit de_budget
        # Roughly popsize * maxiter calls.
        # 12 * 25 = 300 calls. If budget is 500, de_budget is 250. So maxiter ~ 20.
        maxiter_de = max(1, de_budget // 12)
        
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5.0, 5.0)] * dim,
                popsize=12,
                maxiter=maxiter_de,
                seed=42,
                tol=1e-8,
                init='latinhypercube',
                polish=False,  # We will do our own polishing
                update_interval=1
            )
            
        except Exception:
            pass

        if best_x is None:
            # Fallback if DE failed completely
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        # Phase 2: Multi-Start Local Polish using L-BFGS-B
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock once in the valley.
        # We will restart L-BFGS-B from the best point found, and potentially from a few
        # other promising points if we have budget left.
        
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0 and time.time() < deadline - 5.0:
            # First, polish the global best
            try:
                res_lbfgsb = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15}
                )
            except Exception:
                pass

        # Check if we have budget for more restarts
        # We can try to restart from points that were near the best but not quite.
        # However, tracking history is complex. Instead, let's just do a few random
        # local searches around the current best if time/budget permits, or simply
        # continue L-BFGS-B if it stopped early due to iteration limit but not convergence.
        
        # A better strategy: If L-BFGS-B stopped due to maxiter, we can restart it
        # from the new best point.
        
        # Let's check if we have significant budget left.
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 5.0:
             # Try a few more L-BFGS-B restarts from the current best
             # This helps if the first run hit maxiter.
             for _ in range(3):
                 if calls_made >= budget or time.time() > deadline - 2.0:
                     break
                 try:
                     res_lbfgsb2 = minimize(
                         wrapper,
                         best_x,
                         method='L-BFGS-B',
                         bounds=[(-5.0, 5.0)] * dim,
                         options={'maxiter': 200, 'gtol': 1e-15, 'ftol': 1e-15}
                     )
                 except Exception:
                     break

        # Phase 3: Final Nelder-Mead or Simple Local Search if budget remains
        # Sometimes L-BFGS-B stops early due to gtol but can still improve with small steps.
        # Or use a few random perturbations around best_x.
        if calls_made < budget and time.time() < deadline - 2.0:
            # Try a few random steps around the best point
            # Use a decreasing step size
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01]
            for step_size in step_sizes:
                if calls_made >= budget or time.time() > deadline - 1.0:
                    break
                for _ in range(5):
                    if calls_made >= budget or time.time() > deadline - 1.0:
                        break
                    perturbation = np.random.uniform(-step_size, step_size, size=dim)
                    x_new = best_x + perturbation
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = wrapper(x_new)
                    if f_new < best_f:
                        # If improved, do a quick polish
                        if calls_made < budget - 5 and time.time() < deadline - 2.0:
                            try:
                                minimize(
                                    wrapper,
                                    best_x,
                                    method='L-BFGS-B',
                                    bounds=[(-5.0, 5.0)] * dim,
                                    options={'maxiter': 50, 'gtol': 1e-15, 'ftol': 1e-15}
                                )
                            except Exception:
                                pass

        return best_x, best_f
