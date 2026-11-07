"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # We will use a subset of the budget for multi-start L-BFGS-B
        # and keep some for final local refinement.
        
        # Strategy:
        # 1. Use several random starting points.
        # 2. For each, run L-BFGS-B with a limited number of function evaluations.
        # 3. Track the best point found.
        # 4. Use remaining budget for a final, more thorough local search from the best point.
        
        # L-BFGS-B uses numerical gradients, so each gradient evaluation costs (dim+1) function calls.
        # A typical L-BFGS-B iteration might take a few line searches, each costing a few function evals.
        # Let's estimate ~10-20 function evals per L-BFGS-B iteration.
        # With dim=5, gradient costs 6 calls.
        
        # Let's allocate budget:
        # - 50% for multi-start exploration (several L-BFGS-B runs from different starts)
        # - 50% for final refinement
        
        # Actually, L-BFGS-B can be very efficient. Let's try a different approach:
        # Use Nelder-Mead or L-BFGS-B from multiple starts, but carefully manage budget.
        
        # A safer approach: use a population-based method like CMA-ES or a simple
        # multi-start with local optimization.
        
        # Let's use scipy.optimize.minimize with method='L-BFGS-B' from multiple random starts.
        # Each call to minimize can use up to 'maxiter' iterations.
        # We need to count function evaluations precisely.
        
        # Alternative: Implement a simple but effective multi-start local search.
        # Start with several random points, do local L-BFGS-B from each, keep best.
        
        # Let's use a wrapper to count function evaluations.
        call_count = [0]
        
        def counting_func(x):
            call_count[0] += 1
            return func(x)
        
        # Strategy:
        # 1. First, do a quick random search to find a few good starting points.
        # 2. Then run L-BFGS-B from the best few points.
        
        # Phase 1: Random search to find good initial points
        # Use 20% of budget for random search
        random_budget = max(10, budget // 5)
        
        best_x = None
        best_f = np.inf
        
        for i in range(random_budget):
            if call_count[0] >= budget - 10:  # Leave some for refinement
                break
            x = np.random.uniform(-5, 5, dim)
            f = counting_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local refinement from best point using L-BFGS-B
        # Use remaining budget for L-BFGS-B
        remaining = budget - call_count[0]
        
        if remaining > 20 and best_x is not None:
            try:
                # L-BFGS-B with maxiter controlled by remaining budget
                # Each iteration costs ~ (dim + 1) * line_search_evals
                # Let's be conservative
                maxiter = max(5, remaining // 20)
                
                result = minimize(
                    counting_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: If we have budget left, do another round of local search
        # from slightly perturbed versions of the best point
        remaining = budget - call_count[0]
        if remaining > 10 and best_x is not None:
            # Try a few perturbations
            n_perturb = min(remaining // 15, 5)
            for i in range(n_perturb):
                if call_count[0] >= budget - 5:
                    break
                # Small perturbation
                sigma = 0.5 / (i + 1)
                x_pert = best_x + sigma * np.random.standard_normal(dim)
                x_pert = np.clip(x_pert, -5, 5)
                
                # Quick local search from this point
                remaining_sub = budget - call_count[0] - 5
                if remaining_sub > 10:
                    try:
                        result = minimize(
                            counting_func,
                            x_pert,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': max(3, remaining_sub // 25),
                                'ftol': 1e-10,
                                'gtol': 1e-6,
                                'maxls': 10
                            }
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
