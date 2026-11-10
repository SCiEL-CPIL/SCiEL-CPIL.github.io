"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_x = np.zeros(dim)
        best_fx = np.inf
        
        def evaluate(x):
            nonlocal calls_used, best_x, best_fx
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            calls_used += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol
        n_global = 64
        try:
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            samples = sobol.random(n_global)
        except Exception:
            samples = rng.uniform(size=(n_global, dim))
        
        samples = samples * 10.0 - 5.0
        
        for x in samples:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = evaluate(x)
            if np.isinf(fx):
                break
        
        if best_fx == np.inf:
            evaluate(np.zeros(dim))
            
        x0 = best_x.copy()
        fx0 = best_fx

        # Phase 2: Nelder-Mead
        # Simplex initialization
        # Start with x0 and perturbations
        delta = 0.1
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate simplex
        simplex_f = [evaluate(x) for x in simplex]
        
        # If any evaluation failed, fallback
        if any(np.isinf(f) for f in simplex_f):
            return best_x, best_fx

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        tol = 1e-8
        max_iter = 100
        
        for _ in range(max_iter):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Sort simplex by f
            sorted_idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in sorted_idx]
            simplex_f = [simplex_f[i] for i in sorted_idx]
            
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < tol:
                break
            if np.max(np.std(simplex, axis=0)) < 1e-5:
                break

            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            x_best = simplex[0]
            f_best = simplex_f[0]
            x_second = simplex[1]
            f_second = simplex_f[1]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            x_r = centroid + alpha * (centroid - x_worst)
            # Clip to bounds
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if np.isinf(f_r):
                break
                
            if f_r < f_second:
                # Expand
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                
                if np.isinf(f_e):
                    # Reject expansion
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
            else:
                if f_r < f_worst:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contract
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    
                    if np.isinf(f_c):
                         # Shrink
                         for i in range(1, dim + 1):
                             simplex[i] = x_best + sigma * (simplex[i] - x_best)
                             simplex_f[i] = evaluate(simplex[i])
                         break # If shrink fails, likely stuck or out of budget
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex_f[i] = evaluate(simplex[i])
                            if np.isinf(simplex_f[i]):
                                break
                        if any(np.isinf(f) for f in simplex_f):
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
