import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 290  # 290s safety margin out of 300s
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def nelder_mead(x0, f0, max_iter):
            nonlocal best_x, best_f, calls
            
            n = dim
            # Initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Initialize simplex points
            for i in range(n):
                x_i = x0.copy()
                # Use a step size of 0.5 relative to the coordinate
                step = 0.5 if x0[i] == 0 else 0.5 * np.sign(x0[i])
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                fvals.append(eval_func(x_i))
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            stagnation_count = 0
            best_in_nm = fvals[0]
            
            for _ in range(max_iter):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                
                # Check for stagnation
                if fvals[0] < best_in_nm - 1e-10:
                    best_in_nm = fvals[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 30:
                        break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])
                
                # Re-sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
            
            return simplex[0], fvals[0]

        def one_plus_one_es(x0, fx0, max_iter):
            nonlocal best_x, best_f, calls
            
            x = x0.copy()
            fx = fx0
            sigma = 0.5
            no_improve = 0
            
            for _ in range(max_iter):
                if time.time() > self.deadline:
                    break
                if calls >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.85
                    if no_improve > 20:
                        break
                
                sigma = np.clip(sigma, 0.01, 5.0)
            
            return x, fx

        # Phase 1: Multi-start exploration
        # Start at origin and 2 random points
        starts = [np.zeros(dim)]
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, dim))
        
        best_start_x = None
        best_start_f = np.inf
        
        for x0 in starts:
            if calls >= budget or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)
            if f0 < best_start_f:
                best_start_f = f0
                best_start_x = x0.copy()
        
        if best_start_x is None:
            best_start_x = np.zeros(dim)
            best_start_f = eval_func(best_start_x)
        
        # Phase 2: Nelder-Mead refinement
        # Allocate 70% of remaining budget to NM
        remaining = budget - calls
        nm_budget = int(remaining * 0.7)
        
        if nm_budget > 10:
            x_nm, f_nm = nelder_mead(best_start_x, best_start_f, nm_budget)
        
        # Phase 3: (1+1)-ES Polish
        # Use remaining budget for ES polish from the best point found
        es_budget = budget - calls
        if es_budget > 10:
            x_es, f_es = one_plus_one_es(best_x, best_f, es_budget)
        
        return best_x, best_f
