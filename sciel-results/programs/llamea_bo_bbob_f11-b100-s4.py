import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        x_best = None
        f_best = np.inf
        calls_made = 0
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search via Random Sampling
        # Use a small fraction of the budget to find a good starting point
        n_initial = min(50, self.budget // 10)
        if n_initial < 5:
            n_initial = 5
            
        # Ensure we don't exceed budget
        n_initial = min(n_initial, self.budget)
        
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
                
        if x_best is None:
            # Fallback if no evaluation happened
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            calls_made += 1
            
        # Phase 2: Local Search via Nelder-Mead
        # We run Nelder-Mead from the best candidate.
        # If it converges early, we run a second, finer-grained Nelder-Mead.
        
        def run_nelder_mead(x_start, f_start, step_size, max_calls):
            nonlocal calls_made, f_best, x_best
            
            # Initialize simplex
            simplex = [x_start.copy()]
            f_simplex = [f_start]
            
            for i in range(self.dim):
                if calls_made >= self.budget or time.time() > deadline:
                    return f_best, x_best
                x_i = x_start.copy()
                if abs(x_i[i]) < 1e-8:
                    x_i[i] = step_size
                else:
                    x_i[i] += step_size
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls_made += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                
                if f_i < f_best:
                    f_best = f_i
                    x_best = x_i.copy()

            # Nelder-Mead Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            # Iterate until budget or time limit
            while calls_made < self.budget and (self.budget - calls_made) <= max_calls:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                
                if calls_made >= self.budget:
                    break
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < f_simplex[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if calls_made < self.budget:
                        f_e = func(x_e)
                        calls_made += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                            if f_e < f_best:
                                f_best = f_e
                                x_best = x_e.copy()
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                            if f_r < f_best:
                                f_best = f_r
                                x_best = x_r.copy()
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        if f_r < f_best:
                            f_best = f_r
                            x_best = x_r.copy()
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    if f_r < f_best:
                        f_best = f_r
                        x_best = x_r.copy()
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                        
                    x_c = np.clip(x_c, -5, 5)
                    
                    if calls_made < self.budget:
                        f_c = func(x_c)
                        calls_made += 1
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                            if f_c < f_best:
                                f_best = f_c
                                x_best = x_c.copy()
                        else:
                            # Shrinkage
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls_made < self.budget:
                                    f_shrink = func(simplex[i])
                                    calls_made += 1
                                    f_simplex[i] = f_shrink
                                    if f_shrink < f_best:
                                        f_best = f_shrink
                                        x_best = simplex[i].copy()
                                else:
                                    break
                            if calls_made >= self.budget:
                                break
                
                # Convergence check
                # If the range of function values is very small, stop
                if max(f_simplex) - min(f_simplex) < 1e-10:
                    break
                # If the simplex is very small, stop
                if np.max(np.abs(np.array(simplex) - x_best)) < 1e-8:
                    break
                
            return f_best, x_best

        # First Nelder-Mead run with a standard step size
        # Allocate most of the remaining budget to this run
        initial_step = 1.0
        remaining = self.budget - calls_made
        # Reserve a bit for potential second run
        budget_for_nm1 = int(remaining * 0.8)
        
        run_nelder_mead(x_best, f_best, initial_step, budget_for_nm1)
        
        # If we have significant budget left, run a second, finer Nelder-Mead
        if calls_made < self.budget and time.time() < deadline:
            remaining_after_nm1 = self.budget - calls_made
            if remaining_after_nm1 > 20: # Minimum threshold to be useful
                # Use a smaller step size for the second run to refine
                second_step = 0.2
                run_nelder_mead(x_best, f_best, second_step, remaining_after_nm1)

        return x_best, f_best
