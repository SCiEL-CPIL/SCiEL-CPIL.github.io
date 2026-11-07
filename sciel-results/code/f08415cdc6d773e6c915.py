"""Direct approach for BBOB f22 (Gallagher 21 peaks) optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Multi-start Differential Evolution + L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 280s to allow for cleanup/return
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Initial Broad Exploration
        # We use a small set of initial points to ensure we don't start from zero,
        # but we primarily rely on DE for the heavy lifting.
        
        # Add center and a few structured points to help DE initialization if it struggles
        initial_hints = [np.zeros(dim)]
        for d in range(dim):
            for v in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_hints.append(pt)
        
        for x in initial_hints:
            if not time_ok():
                break
            eval_and_track(x)

        # Phase 2: Differential Evolution (Global Search)
        # DE is effective for multimodal functions like f22.
        # We limit the population size and iterations to fit within the budget.
        
        # Calculate remaining budget for DE
        remaining_budget = budget - calls - 10  # Reserve 10 for final polish
        
        if remaining_budget > 20 and time_ok():
            try:
                # Popsize 10 is reasonable for dim=5.
                # Maxiter is dynamic based on budget.
                popsize = 10
                # Each generation costs popsize evaluations.
                # We want to use ~80% of remaining budget for DE.
                max_iter = max(1, int((remaining_budget * 0.8) / popsize))
                
                # Differential Evolution wrapper
                # Note: DE handles bounds internally, but we pass them explicitly.
                
                def de_func(x):
                    # DE calls this function. We must track calls.
                    return eval_and_track(x)

                bounds = [(-5.0, 5.0)] * dim
                
                # Run DE
                # seed ensures reproducibility if needed, but we use rng elsewhere
                res_de = differential_evolution(
                    de_func,
                    bounds=bounds,
                    popsize=popsize,
                    maxiter=max_iter,
                    seed=42,
                    tol=1e-6,
                    polish=False, # We do our own polish
                    init='latinhypercube' # Good initialization for DE
                )
                
                if res_de.success or (res_de.fun < best_f):
                    best_f = res_de.fun
                    best_x = res_de.x
            except Exception:
                pass

        # Phase 3: Local Refinement (Polish)
        # Use L-BFGS-B for fast, gradient-based (numerical) refinement.
        # It's generally more efficient than Nelder-Mead for smooth-ish regions 
        # after a good global search.
        
        if time_ok() and calls < budget - 5:
            try:
                # L-BFGS-B with bounds
                res_lbfgs = minimize(
                    eval_and_track,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxls': 30}
                )
                
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead Polish
        # NM is derivative-free and can handle non-smoothness better if L-BFGS-B stalls.
        if time_ok() and calls < budget:
            try:
                res_nm = minimize(
                    eval_and_track,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
