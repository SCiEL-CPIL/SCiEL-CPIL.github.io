import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Combines Sobol global exploration with a robust local trust-region method (Nelder-Mead).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        # Use a modest number of initial points to find the basin of attraction.
        # Based on observation, 10 points is sufficient and leaves more budget for local search.
        from scipy.stats import qmc
        n_init = min(10, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        init_points = sampler.random(n=n_init)
        init_points = 2.0 * init_points - 1.0
        init_points = 5.0 * init_points
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(init_points[i])
            
        if best_x is None:
            best_x = np.zeros(dim)
            eval_pt(best_x)

        # Phase 2: Local refinement using Nelder-Mead
        # The Schaffers F7 function is ill-conditioned but has a clear local minimum.
        # Nelder-Mead is robust and derivative-free, suitable for this landscape.
        # We run it with a high iteration limit to allow for slow convergence on ill-conditioned problems.
        from scipy.optimize import minimize
        
        # Set up Nelder-Mead with tight tolerances
        # Initial simplex size is handled by scipy, but we can ensure it starts from the best Sobol point.
        x0 = best_x.copy()
        
        # We need to be careful with the number of function evaluations.
        # Nelder-Mead in SciPy does not directly expose a 'budget' in the same way, 
        # but we can control it via maxiter and by checking time/budget in a wrapper if necessary.
        # However, since we have a global `calls` counter and `deadline`, we can just let it run 
        # until it converges or we hit the budget. 
        # To prevent it from running away, we can use a wrapper that checks the budget.
        
        def nm_func(x):
            return eval_pt(x)
            
        # Use 'Nelder-Mead' with adaptive step size.
        # 'xatol' and 'fatol' control convergence.
        # 'maxiter' is a safety cap.
        # Since we have a global budget, we rely on the eval_pt checks.
        
        try:
            res = minimize(nm_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5.0, 5.0)
        except Exception:
            pass

        # Phase 3: If budget remains, try a second local refinement from a slightly perturbed best point
        # This helps if the first NM got stuck in a local valley.
        if calls < budget and time.time() < deadline and best_x is not None:
            # Perturb the best point slightly
            perturb = np.random.normal(0, 0.1, dim)
            x0_2 = np.clip(best_x + perturb, -5.0, 5.0)
            
            try:
                res2 = minimize(nm_func, x0_2, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = np.clip(res2.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
