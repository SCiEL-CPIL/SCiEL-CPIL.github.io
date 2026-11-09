import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0 # Safety margin
        
        x_best = None
        f_best = np.inf
        
        # Phase 1: Global Search via Random Sampling
        # Use a small fraction of the budget to find a good starting point
        n_initial = max(10, self.budget // 10)
        n_initial = min(n_initial, self.budget // 2) # Reserve at least half for local search
        
        rng = np.random.default_rng(42)
        
        candidates = []
        for i in range(n_initial):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            candidates.append((f, x))
                
        if x_best is None:
            # Fallback if no evaluation happened
            x_best = np.zeros(self.dim)
            f_best = func(x_best)
            candidates.append((f_best, x_best))
            
        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-Start Nelder-Mead
        # We will run Nelder-Mead from the top few candidates
        # Each run gets a portion of the remaining budget
        
        calls_made = n_initial
        remaining_budget = self.budget - calls_made
        
        # Number of restarts
        n_restarts = 3
        budget_per_restart = remaining_budget // n_restarts
        
        # Ensure at least some budget for each restart if possible
        if budget_per_restart < 10:
             n_restarts = 1
             budget_per_restart = remaining_budget

        for r in range(n_restarts):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Choose starting point
            if r < len(candidates):
                x0 = candidates[r][1].copy()
                f0 = candidates[r][0]
            else:
                # If we run out of distinct candidates, perturb the best
                x0 = x_best + rng.uniform(-0.5, 0.5, self.dim)
                x0 = np.clip(x0, -5, 5)
                f0 = func(x0)
                calls_made += 1
                if f0 < f_best:
                    f_best = f0
                    x_best = x0.copy()
            
            # Nelder-Mead Setup
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            # Initial step size
            step_size = 0.5
            
            for i in range(self.dim):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                x_i = x0.copy()
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

            # Nelder-Mead Iteration
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            # Limit iterations for this restart
            max_iter = budget_per_restart
            
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
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
                    else:
                        break
                
                # Convergence check
                if max(f_simplex) - min(f_simplex) < 1e-10:
                    break
                if np.max(np.abs(np.array(simplex) - x_best)) < 1e-8:
                    break

        return x_best, f_best
