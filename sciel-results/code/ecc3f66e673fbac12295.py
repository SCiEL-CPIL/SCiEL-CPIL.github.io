"""New program: Multi-Start Nelder-Mead optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Nelder-Mead continuous optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        lb = -5.0
        ub = 5.0
        
        # Helper to reflect points back into bounds
        def reflect(x):
            x = np.array(x)
            for i in range(len(x)):
                if x[i] < lb:
                    x[i] = 2 * lb - x[i]
                elif x[i] > ub:
                    x[i] = 2 * ub - x[i]
            return x

        # Nelder-Mead implementation
        def nelder_mead(start_x, max_calls):
            # Initialize simplex
            x0 = np.array(start_x)
            f0 = func(x0)
            calls = 1
            
            # Create simplex
            simplex = [x0]
            simplex_f = [f0]
            
            # Initialize other points with small step
            step = 0.1
            for i in range(n):
                xi = x0.copy()
                # If x0 is close to bound, step inwards
                if x0[i] > ub - step:
                    xi[i] -= step
                elif x0[i] < lb + step:
                    xi[i] += step
                else:
                    xi[i] += step
                simplex.append(xi)
                simplex_f.append(func(xi))
                calls += 1
            
            # Sort simplex
            def sort_simplex():
                nonlocal simplex, simplex_f
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
            
            sort_simplex()
            best_x = simplex[0].copy()
            best_f = simplex_f[0]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            while calls < max_calls:
                # Get worst, second worst, best
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_second = simplex[-2]
                f_second = simplex_f[-2]
                x_best = simplex[0]
                f_best = simplex_f[0]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = reflect(x_r)
                f_r = func(x_r)
                calls += 1
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = reflect(x_e)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < f_second:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - x_worst)
                    
                    x_c = reflect(x_c)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if calls >= max_calls:
                                break
                        if calls >= max_calls:
                            break
                
                sort_simplex()
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Check for convergence
                if np.std(simplex_f) < 1e-8:
                    break
                    
            return best_x, best_f, calls

        # Phase 1: Initial random sampling
        init_calls = 10
        if budget < init_calls + 100: # Fallback for very small budget
            # Just random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = rng.uniform(lb, ub, size=n)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f

        candidates = []
        for _ in range(init_calls):
            x = rng.uniform(lb, ub, size=n)
            f = func(x)
            candidates.append((f, x))
        
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-start Nelder-Mead
        # Allocate budget to top 3 candidates
        remaining_budget = budget - init_calls
        
        # We want to spend the remaining budget on local search.
        # Let's pick top 3.
        num_starts = min(3, len(candidates))
        
        # Distribute budget. Give more to better candidates.
        # Weights: 1.5, 1.0, 0.5
        weights = [1.5, 1.0, 0.5][:num_starts]
        total_weight = sum(weights)
        
        best_overall_x = candidates[0][1]
        best_overall_f = candidates[0][0]
        
        for i in range(num_starts):
            f_init, x_init = candidates[i]
            # Calculate budget for this start
            budget_for_start = int(remaining_budget * (weights[i] / total_weight))
            # Ensure at least 10 calls for NM to initialize simplex
            if budget_for_start < 10:
                budget_for_start = 10
            
            # Don't exceed remaining actual budget
            if budget_for_start > remaining_budget:
                budget_for_start = remaining_budget
            
            if budget_for_start <= 0:
                break
                
            x_best, f_best, calls_used = nelder_mead(x_init, budget_for_start)
            
            if f_best < best_overall_f:
                best_overall_f = f_best
                best_overall_x = x_best
                
            remaining_budget -= calls_used
            
            if remaining_budget <= 0:
                break
                
        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
