import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        x_best = None
        fx_best = float('inf')
        
        # Evaluation counter
        n_evals = 0
        
        def safe_call(x):
            nonlocal n_evals, x_best, fx_best
            if n_evals >= budget:
                return fx_best
            n_evals += 1
            try:
                fx = func(x)
            except Exception:
                return fx_best
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
            return fx
        
        # --- Phase 1: Global Search via Lattice Sampling ---
        # Rastrigin has periodicity of 2. We sample on a grid to hit potential basins.
        # Grid points: -4, -2, 0, 2, 4 (5 points per dim -> 5^5 = 3125 points)
        # But budget is 500. So we can't sample all.
        # We will sample a subset or use a smarter strategy.
        # Let's use a dense random search first, but biased towards grid points?
        # Or just random search.
        
        # Actually, for Rastrigin, the global minimum is at the origin (shifted/rotated).
        # The periodic structure means there are many local minima.
        # A good strategy is to sample widely, then refine the best few.
        
        n_explore = 100
        n_refine = budget - n_explore
        
        # Evaluate center
        x0 = np.zeros(dim)
        fx0 = safe_call(x0)
        
        # Random exploration
        for i in range(1, n_explore):
            if n_evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
        
        # --- Phase 2: Multi-Start Nelder-Mead ---
        # Select top K points from exploration? We don't store all.
        # Instead, let's just restart NM from the best point found, and maybe a few random points.
        
        # But NM is sensitive to initialization.
        # Let's implement a robust NM with restarts.
        
        def nelder_mead(x0, max_iter=50):
            nonlocal n_evals, x_best, fx_best
            if n_evals >= budget:
                return x_best, fx_best
            
            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            fx0_val = safe_call(x0)
            
            for i in range(dim):
                xi = x0.copy()
                if xi[i] < 4.0:
                    xi[i] += 0.5
                else:
                    xi[i] -= 0.5
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                safe_call(xi)
            
            # Sort simplex
            simplex = sorted(simplex, key=lambda x: safe_call(x) if False else 0) # Placeholder, we need stored values
            # Re-evaluate to get values for sorting (or store them)
            # Let's store values in a list of tuples (fx, x)
            
            # Redo NM properly with stored values
            simplex = []
            x_curr = x0.copy()
            fx_curr = safe_call(x_curr)
            simplex.append((fx_curr, x_curr))
            
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if x0[i] < 4.0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fx_i = safe_call(xi)
                simplex.append((fx_i, xi))
            
            # Sort by function value
            simplex.sort(key=lambda t: t[0])
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if n_evals >= budget or time.time() > deadline:
                    break
                
                # Sort
                simplex.sort(key=lambda t: t[0])
                
                # Check convergence
                if simplex[-1][0] - simplex[0][0] < 1e-8:
                    break
                
                # Centroid of all but worst
                x_worst = simplex[-1][1]
                x_worst_fx = simplex[-1][0]
                centroid = np.mean([t[1] for t in simplex[:-1]], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                fx_ref = safe_call(x_ref)
                
                if fx_ref < simplex[0][0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    fx_exp = safe_call(x_exp)
                    if fx_exp < fx_ref:
                        simplex[-1] = (fx_exp, x_exp)
                    else:
                        simplex[-1] = (fx_ref, x_ref)
                elif fx_ref < simplex[-2][0]:
                    simplex[-1] = (fx_ref, x_ref)
                else:
                    # Contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    fx_con = safe_call(x_con)
                    if fx_con < x_worst_fx:
                        simplex[-1] = (fx_con, x_con)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i][1] = simplex[0][1] + sigma * (simplex[i][1] - simplex[0][1])
                            simplex[i][1] = np.clip(simplex[i][1], -5, 5)
                            simplex[i][0] = safe_call(simplex[i][1])
            
            return x_best, fx_best
        
        # Start NM from best point
        if x_best is not None:
            nelder_mead(x_best, max_iter=40)
            
        # If budget remains, do more random restarts + NM
        while n_evals < budget and time.time() < deadline:
            # Pick a random starting point
            x_start = rng.uniform(-5, 5, dim)
            # Quick check: if it's better than current best, run NM
            fx_start = safe_call(x_start)
            if fx_start < fx_best:
                nelder_mead(x_start, max_iter=20)
            else:
                # Maybe just do a few random steps? Or skip.
                # To save budget, just continue searching.
                pass
        
        return x_best, fx_best
