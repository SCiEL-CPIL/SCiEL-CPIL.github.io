import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s limit
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0
        
        def check_time():
            return (time.time() - start_time) > timeout
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Multi-start initialization
        n_starts = 10
        rng = np.random.default_rng(42)
        
        # Generate starts
        starts = []
        for _ in range(n_starts):
            x0 = rng.uniform(-5, 5, d)
            starts.append(x0)
        
        # Evaluate starts
        start_vals = []
        for x0 in starts:
            fx = evaluate(x0)
            start_vals.append(fx)
            
            if check_time() or calls_made >= budget:
                return best_x, best_f

        # Sort starts by function value
        sorted_indices = np.argsort(start_vals)
        
        # 2. Local Optimization with Nelder-Mead
        # We will run NM on the best starts, using remaining budget
        # Budget per NM instance
        remaining_budget = budget - calls_made
        budget_per_nm = max(10, remaining_budget // n_starts)
        
        for idx in sorted_indices:
            if check_time() or calls_made >= budget:
                break
                
            x0 = starts[idx]
            f0 = start_vals[idx]
            
            # Initialize Simplex
            # x0 is the best point, others are x0 + delta * e_i
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(d):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
                
            # Evaluate initial simplex (we already have f0 for x0)
            # Note: evaluate() updates global best, which is fine
            fvals = [f0]
            for i in range(1, d+1):
                fx = evaluate(simplex[i])
                fvals.append(fx)
                
                if check_time() or calls_made >= budget:
                    return best_x, best_f

            # Nelder-Mead Loop
            max_nm_iter = 50
            for _ in range(max_nm_iter):
                if check_time() or calls_made >= budget:
                    break
                    
                # Sort simplex by f value
                sort_idx = np.argsort(fvals)
                simplex = [simplex[i] for i in sort_idx]
                fvals = [fvals[i] for i in sort_idx]
                
                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-8:
                    break
                
                # Centroid of best N points
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                xr = centroid + 1.0 * (centroid - worst)
                fr = evaluate(xr)
                
                if check_time() or calls_made >= budget:
                    return best_x, best_f

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (xr - centroid)
                    fe = evaluate(xe)
                    if check_time() or calls_made >= budget:
                        return best_x, best_f
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = evaluate(xc)
                        if check_time() or calls_made >= budget:
                            return best_x, best_f
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d+1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])
                                if check_time() or calls_made >= budget:
                                    return best_x, best_f
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (worst - centroid)
                        fc = evaluate(xc)
                        if check_time() or calls_made >= budget:
                            return best_x, best_f
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, d+1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])
                                if check_time() or calls_made >= budget:
                                    return best_x, best_f

        return best_x, best_f
