import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f9 (rotated Rosenbrock).
    Combines Differential Evolution for global exploration with L-BFGS-B for local refinement.
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
        # popsize: 15 (good for 5D)
        # maxiter: estimated to fit de_budget
        # Roughly popsize * maxiter calls.
        # 15 * 16 = 240 calls. If budget is 500, de_budget is 250.
        maxiter_de = max(1, de_budget // 15)
        
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5.0, 5.0)] * dim,
                popsize=15,
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

        # Phase 2: Local Polish using L-BFGS-B with multiple restarts
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock once in the valley.
        # It uses gradient approximations (finite differences), costing ~dim+1 calls per step.
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0 and time.time() < deadline - 10.0:
            # First L-BFGS-B run from the best point
            try:
                res_lbfgsb = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-12}
                )
            except Exception:
                pass

        # Phase 3: Multiple local restarts with perturbed starting points
        # This helps escape flat regions in the Rosenbrock valley
        if calls_made < budget and time.time() < deadline - 5.0:
            restarts = 5
            for r in range(restarts):
                if calls_made >= budget or time.time() > deadline - 2.0:
                    break
                # Create perturbed starting point
                step_size = 0.5 * (1.0 - r / restarts)  # Decreasing step size
                perturbation = np.random.uniform(-step_size, step_size, size=dim)
                x_start = best_x + perturbation
                x_start = np.clip(x_start, -5.0, 5.0)
                
                try:
                    res_restart = minimize(
                        wrapper,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-10}
                    )
                except Exception:
                    pass

        # Phase 4: Final fine-tuning with very small steps
        if calls_made < budget and time.time() < deadline - 1.0:
            # Adaptive random search with decreasing step size
            n_attempts = min(50, budget - calls_made)
            for i in range(n_attempts):
                if calls_made >= budget or time.time() > deadline - 0.5:
                    break
                # Decreasing step size
                step_size = 0.1 * (1.0 - i / max(1, n_attempts))
                if step_size < 0.001:
                    step_size = 0.001
                perturbation = np.random.uniform(-step_size, step_size, size=dim)
                x_new = best_x + perturbation
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = wrapper(x_new)

        return best_x, best_f
