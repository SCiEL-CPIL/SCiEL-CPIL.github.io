# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start strategy
        n_starts = 3
        base_budget = budget // n_starts
        budgets = [base_budget] * n_starts
        budgets[-1] += budget - base_budget * n_starts

        for start_budget in budgets:
            if calls_made >= budget:
                break
            
            # Phase 1: Random Sampling
            n_random = max(5, start_budget // 5)
            for _ in range(n_random):
                if calls_made >= budget:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                safe_call(x)

            # Phase 2: Nelder-Mead
            x0 = best_x
            if x0 is None:
                x0 = np.zeros(dim)
            
            # Initialize simplex
            step = 1.0
            simplex = []
            for i in range(dim + 1):
                xi = x0.copy()
                if i < dim:
                    xi[i] += step
                simplex.append(xi)
            
            # Evaluate simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                fvals[i] = safe_call(simplex[i])
            
            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Loop
            while calls_made < budget:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]
                
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_best = simplex[0]
                f_best = fvals[0]
                
                # Convergence check
                if f_worst - f_best < 1e-8:
                    break
                if np.max(np.abs(simplex - x_best)) < 1e-6:
                    break

                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_call(x_ref)
                
                if f_ref < f_best:
                    # Expansion
                    if calls_made < budget:
                        x_exp = x_centroid + gamma * (x_ref - x_centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = safe_call(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Replace worst
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_call(x_con)
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if calls_made < budget:
                                fvals[i] = safe_call(simplex[i])
                            else:
                                break
                        if calls_made >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
