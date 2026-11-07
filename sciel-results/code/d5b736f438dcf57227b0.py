import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, dim, deadline):
        """Multistart Nelder-Mead local optimizer."""
        best_f = func(x0)
        best_x = x0.copy()
        calls = 1
        
        # Nelder-Mead simplex
        alpha = 1.5  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        # Initialize simplex
        simplex = [x0.copy() for _ in range(dim + 1)]
        fvals = [best_f]
        
        # Create initial simplex with small perturbations
        for i in range(dim):
            simplex[i] = x0.copy()
            simplex[i][i] += 0.1
            fvals.append(func(simplex[i]))
            calls += 1
        
        max_iter = budget - calls
        for iteration in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if fvals[0] <= f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            elif f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
            
            # Track best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        # Leave 10 seconds margin for safety
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1
        
        # Strategy: Multistart Nelder-Mead with CMA-ES-like global exploration
        # Phase 1: Quick global exploration with random points
        n_global = min(30, budget // 10)
        for i in range(n_global):
            if calls >= budget or not time_ok():
                break
            x_rand = np.random.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            calls += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()
        
        # Phase 2: Multistart Nelder-Mead from multiple starting points
        # Use the best points found so far as starting points
        n_starts = min(8, (budget - calls) // 50)
        
        # Collect starting points: best point + random perturbations
        start_points = [best_x.copy()]
        for i in range(n_starts - 1):
            if calls >= budget or not time_ok():
                break
            # Perturb best point
            x_pert = best_x + np.random.normal(0, 1.0, dim)
            x_pert = np.clip(x_pert, -5, 5)
            start_points.append(x_pert)
        
        # Run Nelder-Mead from each starting point
        remaining_budget = budget - calls
        budget_per_start = max(30, remaining_budget // len(start_points))
        
        for i, x_start in enumerate(start_points):
            if calls >= budget or not time_ok():
                break
            
            # Allocate budget: more for later starts if budget allows
            alloc = min(budget_per_start, budget - calls)
            if alloc < 10:
                break
            
            x_opt, f_opt, used = self._nelder_mead(func, x_start, alloc, dim, deadline)
            calls += used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
            
            # If we found a significantly better point, restart NM from there
            if f_opt < best_f + 1e-6 and i < len(start_points) - 1:
                # Update subsequent start points based on new best
                for j in range(i + 1, len(start_points)):
                    start_points[j] = best_x + np.random.normal(0, 0.5, dim)
                    start_points[j] = np.clip(start_points[j], -5, 5)
        
        # Phase 3: Final local refinement with CMA-ES-like small steps
        if calls < budget and time_ok():
            # Use remaining budget for fine-tuning around best point
            remaining = budget - calls
            sigma = 0.1
            for i in range(remaining):
                if not time_ok() or calls >= budget:
                    break
                x_try = best_x + sigma * np.random.normal(0, 1, dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    sigma *= 0.95  # Shrink step size
                else:
                    sigma *= 0.99
        
        if best_x is None:
            best_x = np.zeros(dim)
            if calls < budget:
                best_f = func(best_x)
                calls += 1
        
        return best_x, best_f
