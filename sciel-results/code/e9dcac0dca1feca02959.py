import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Global Search using Latin Hypercube Sampling (LHS)
        # LHS provides better space-filling properties than uniform random.
        # We allocate ~40% of the budget to global search to find the basin of the global optimum.
        # This leaves more budget for the intensive local search phase.
        n_global = int(budget * 0.4)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        # Sort by fitness
        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # Phase 2: Local Refinement using (1+1)-ES
        # We take the top K candidates from the global search and run (1+1)-ES on them.
        # (1+1)-ES is robust, derivative-free, and works well for BBOB-21.
        # We allocate the remaining budget to local search.
        
        n_local_candidates = 5
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        # (1+1)-ES needs enough steps to converge. 
        # We'll use a simple step-size adaptation (1/5th rule or similar).
        # For dim=5, 50-200 calls per local search is reasonable.
        budget_per_local = max(20, remaining_budget // n_local_candidates)
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best global sample
            x = lhs[sorted_indices[k]].copy()
            f = f_values[sorted_indices[k]]
            
            # If this starting point is already worse than the global best,
            # we might still find a better local optimum in its basin,
            # but we prioritize the best ones.
            
            # (1+1)-ES parameters
            # Initial step size: large enough to explore, small enough to converge.
            # The box is [-5, 5]^5, so a step size of ~0.5 is reasonable initially.
            step_size = 0.5
            sigma = step_size
            n_evals_local = 0
            max_local_evals = budget_per_local
            
            # 1/5th rule parameters
            success_rate_target = 0.2
            sigma_increase = 1.1
            sigma_decrease = 0.9
            window_size = 10
            successes = 0
            window = []
            
            while n_evals_local < max_local_evals and calls < budget and time.time() < deadline:
                # Generate mutant
                mutation = rng.normal(0, sigma, size=dim)
                x_new = x + mutation
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate mutant
                f_new = func(x_new)
                calls += 1
                n_evals_local += 1
                
                # Update window for 1/5th rule
                if f_new < f:
                    successes += 1
                    window.append(1)
                    # Accept
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)
                
                if len(window) >= window_size:
                    success_rate = np.mean(window[-window_size:])
                    if success_rate > success_rate_target:
                        sigma *= sigma_increase
                    elif success_rate < success_rate_target:
                        sigma *= sigma_decrease
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Check for convergence
                if sigma < 1e-8:
                    break

        return best_x, best_f
