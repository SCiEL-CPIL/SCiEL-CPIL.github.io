import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        # Multi-start Nelder-Mead
        # We will try several restarts.
        # Each restart gets a sub-budget.
        # If we run out of total budget, we stop.
        
        max_restarts = 10
        sub_budget = 50 # 10 * 50 = 500
        
        for _ in range(max_restarts):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Random start
            x = rng.uniform(-5, 5, self.dim)
            
            # Run Nelder-Mead
            x, f, used_calls = self._nelder_mead(func, x, calls, self.budget - calls, sub_budget, rng)
            calls += used_calls
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            # If we made significant progress, maybe continue?
            # If we used all sub_budget, restart.
            # If we converged early, restart with remaining budget.
            
        return best_x, best_f

    def _nelder_mead(self, func, x0, global_calls, remaining_total, sub_budget, rng):
        """
        Runs Nelder-Mead starting from x0.
        Returns best_x, best_f, calls_used.
        """
        n = self.dim
        calls = 0
        max_calls = min(remaining_total, sub_budget)
        
        # Initialize simplex
        # x0 is the best point of the simplex initially? 
        # Standard NM: x0 is one vertex, others are x0 + delta * e_i
        delta = 0.5
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(n):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += delta
            
        # Evaluate simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            fvals[i] = func(simplex[i])
            calls += 1
            if calls > max_calls:
                break
                
        # Find best, worst, second worst
        idx = np.argsort(fvals)
        best_idx = idx[0]
        worst_idx = idx[-1]
        second_worst_idx = idx[-2]
        
        best_f = fvals[best_idx]
        
        # Coefficients
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < max_calls:
            # Reorder simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            best_idx = 0
            worst_idx = n
            second_worst_idx = n - 1
            
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-8:
                break
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[best_idx]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    fvals[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    fvals[worst_idx] = f_r
            elif f_r < fvals[second_worst_idx]:
                # Accept reflection
                simplex[worst_idx] = x_r
                fvals[worst_idx] = f_r
            else:
                # Contraction
                if f_r < fvals[worst_idx]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, fvals[worst_idx]):
                    simplex[worst_idx] = x_c
                    fvals[worst_idx] = f_c
                else:
                    # Shrinkage
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls > max_calls:
                            break
                    if calls > max_calls:
                        break
                        
            # Update best
            current_best_f = np.min(fvals)
            if current_best_f < best_f:
                best_f = current_best_f
                best_idx = np.argmin(fvals)
                
        best_x = simplex[np.argmin(fvals)]
        return best_x, best_f, calls
