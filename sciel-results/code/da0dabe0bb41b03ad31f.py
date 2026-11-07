import numpy as np
import time
import math
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            try:
                f_val = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(f_val):
                return best_f
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # Use 40% of budget for DE to find a good starting point
        de_budget = int(budget * 0.4)
        
        # Adjust popsize based on budget to be safe on calls
        # popsize=10 -> 50 individuals.
        # Init adds 50 evals.
        # 1 iteration adds 50 evals.
        # Total ~100 calls.
        
        try:
            res_de = differential_evolution(
                eval_func, 
                bounds=[(-5, 5)] * dim, 
                popsize=10, 
                maxiter=1,
                seed=0,
                tol=1e-6,
                init='latinhypercube'
            )
            # Update best
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass

        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Local Polish with Nelder-Mead
        # Use remaining budget
        # Nelder-Mead is effective for Rosenbrock-like functions
        
        # Initial simplex: best_x and best_x + delta*e_i
        delta = 0.5
        simplex = [best_x.copy()]
        for i in range(dim):
            xi = best_x.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate initial simplex
        simplex_f = []
        for sx in simplex:
            sx = np.clip(sx, -5, 5)
            fx = eval_func(sx)
            simplex_f.append(fx)
        
        # Nelder-Mead iterations
        max_iter = 100
        for it in range(max_iter):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + 1.0 * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = eval_func(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + 2.0 * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = eval_func(xc)
                
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = eval_func(simplex[i])
            
            # Check convergence
            if max(simplex_f) - min(simplex_f) < 1e-10:
                break

        # Phase 3: Optional second round of local polish if budget remains
        if calls_used < budget and time.time() < deadline:
            # Try Powell for further refinement
            try:
                res_powell = minimize(
                    eval_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxiter': 20, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = res_powell.x.copy()
            except Exception:
                pass

        return best_x, best_f
