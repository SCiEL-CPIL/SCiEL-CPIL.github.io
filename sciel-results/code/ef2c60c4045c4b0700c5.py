"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimization with adaptive population and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Initial exploration with a small population
        # Use 15% of budget for initial sampling
        n_initial = max(5, budget // 7)
        population = []
        
        for i in range(n_initial):
            if call_count[0] >= budget - 20:
                break
            # Mix of uniform and slightly biased sampling
            if i % 3 == 0:
                x = np.random.uniform(-5, 5, dim)
            elif i % 3 == 1:
                x = np.random.uniform(-3, 3, dim)
            else:
                x = np.random.uniform(-2, 2, dim)
            x = np.clip(x, lower, upper)
            f = counting_func(x)
            population.append((f, x.copy()))
        
        if not population:
            # Fallback: single random point
            x = np.random.uniform(-5, 5, dim)
            f = counting_func(x)
            population = [(f, x.copy())]
        
        # Sort by fitness
        population.sort(key=lambda t: t[0])
        best_f, best_x = population[0]

        # Phase 2: Local refinement from top candidates
        # Use remaining budget for L-BFGS-B from best few points
        remaining = budget - call_count[0]
        
        # Try L-BFGS-B from the best 3 candidates
        n_refine = min(3, len(population))
        for i in range(n_refine):
            if call_count[0] >= budget - 10:
                break
            
            f_init, x_init = population[i]
            remaining_sub = budget - call_count[0] - 5
            
            if remaining_sub < 15:
                break
            
            try:
                # Estimate max iterations conservatively
                # L-BFGS-B with numerical gradient: ~6 evals per gradient + line search
                # Assume ~15-25 evals per iteration on average
                maxiter = max(3, min(50, remaining_sub // 20))
                
                result = minimize(
                    counting_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 30
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Final fine-tuning with Nelder-Mead or more L-BFGS-B
        remaining = budget - call_count[0]
        if remaining > 10 and best_x is not None:
            # Try Nelder-Mead for final polish
            try:
                result = minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, remaining // 3),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: If still have budget, do a few more L-BFGS-B from perturbed best
        remaining = budget - call_count[0]
        if remaining > 15 and best_x is not None:
            n_perturb = min(3, remaining // 20)
            for i in range(n_perturb):
                if call_count[0] >= budget - 5:
                    break
                sigma = 0.3 / (i + 1)
                x_pert = best_x + sigma * np.random.standard_normal(dim)
                x_pert = np.clip(x_pert, lower, upper)
                
                remaining_sub = budget - call_count[0] - 5
                if remaining_sub < 10:
                    break
                
                try:
                    maxiter = max(3, remaining_sub // 25)
                    result = minimize(
                        counting_func,
                        x_pert,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': maxiter,
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                            'maxls': 20
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
