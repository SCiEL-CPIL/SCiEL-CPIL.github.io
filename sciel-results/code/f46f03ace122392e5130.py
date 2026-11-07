import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(12345)

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            try:
                f = func(x)
                calls_made += 1
                return f
            except:
                return np.inf

        # Phase 1: Global Exploration with 50 random points
        n_initial = 50
        initial_points = []
        initial_values = []
        
        for _ in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            p = rng.uniform(-5, 5, dim)
            f = safe_call(p)
            if f < np.inf:
                initial_points.append(p)
                initial_values.append(f)
                update_best(p, f)
        
        if not initial_points:
            # Fallback if no valid points found
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f >= np.inf:
                best_f = 0.0
            return best_x, best_f

        # Sort by function value and keep top 5
        initial_points = np.array(initial_points)
        initial_values = np.array(initial_values)
        sorted_indices = np.argsort(initial_values)
        top_k = min(5, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        
        # Extract top 5 points and their values
        top_points = initial_points[top_indices]
        top_values = initial_values[top_indices]
        
        # Phase 2: Parallel (1+1)-ES from each of the top 5 starting points
        # Allocate remaining budget evenly to each of the 5 runs
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
        
        budget_per_es = remaining_budget // top_k
        
        # (1+1)-ES parameters
        # Initial step size: heuristic based on problem scale
        # For BBOB f13 (sharp ridge), we need careful step sizing
        # Start with a moderate step size and adapt
        sigma = 0.5  # Initial step size
        success_rate = 0.5  # Target success rate for adaptation
        adaptation_factor = 1.015
        cooling_factor = 0.985
        
        # Run (1+1)-ES from each top point in parallel (sequentially but with separate budgets)
        for i in range(top_k):
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Start from the i-th best point
            current_x = top_points[i].copy()
            current_f = top_values[i]
            
            # If this starting point is already the global best, start from there
            # Otherwise, we'll try to improve from this basin
            if current_f < best_f:
                update_best(current_x, current_f)
            
            # Run (1+1)-ES with allocated budget
            es_calls = 0
            while es_calls < budget_per_es and calls_made < budget and time.time() < deadline:
                # Generate a trial point
                # Use adaptive step size
                trial_x = current_x + sigma * rng.standard_normal(dim)
                trial_x = np.clip(trial_x, -5, 5)
                
                trial_f = safe_call(trial_x)
                es_calls += 1
                
                if trial_f >= np.inf:
                    break
                
                # Adapt step size based on success
                if trial_f < current_f:
                    # Success: move to new point, increase step size slightly
                    current_x = trial_x
                    current_f = trial_f
                    update_best(current_x, current_f)
                    sigma *= adaptation_factor
                    sigma = min(sigma, 5.0)  # Cap step size
                else:
                    # Failure: decrease step size
                    sigma *= cooling_factor
                    sigma = max(sigma, 1e-4)  # Floor step size
                
                # Early stopping if convergence is reached
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
