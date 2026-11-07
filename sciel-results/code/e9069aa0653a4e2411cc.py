import numpy as np
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration using Sobol sequence (or LHS)
        # We use 100 points for global search to get a good initial guess.
        n_global = 100
        if budget <= n_global:
            # If budget is very small, just do random/grid
            # But budget is 500, so this won't happen.
            pass
            
        # Generate Sobol points
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            points = sampler.random(n=n_global)
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
        except ImportError:
            # Fallback to Latin Hypercube or Random
            # Simple LHS
            points = np.zeros((n_global, dim))
            for i in range(dim):
                # Divide interval into n_global bins
                bin_edges = np.linspace(-5, 5, n_global + 1)
                # Pick one random point in each bin
                # Use deterministic random for reproducibility? 
                # The task doesn't require deterministic output, just valid execution.
                # But using a fixed seed inside the optimizer might be good for stability.
                # However, the evaluator uses different instances.
                # Let's use np.random.default_rng(42) for consistent behavior across instances?
                # No, the function is different. The sampling points can be the same.
                rng = np.random.default_rng(42)
                points[:, i] = rng.uniform(bin_edges[:-1], bin_edges[1:])
            # Shuffle to decorrelate dimensions
            # For LHS, we need to permute the rows for each dimension?
            # Standard LHS: 
            # 1. Divide each dimension into N intervals.
            # 2. Select one point uniformly at random from each interval.
            # 3. Shuffle the points in each dimension independently.
            # My simple loop above doesn't shuffle. Let's do it properly.
            points = np.zeros((n_global, dim))
            rng = np.random.default_rng(42)
            for i in range(dim):
                bin_edges = np.linspace(-5, 5, n_global + 1)
                # One point per bin
                pts = rng.uniform(bin_edges[:-1], bin_edges[1:])
                # Shuffle
                rng.shuffle(pts)
                points[:, i] = pts

        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate global points
        # Store top K points for local search
        # We want the top 3
        top_points = []
        
        for x in points:
            if calls_made >= budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Maintain top 3 list
            # Simple insertion sort for small K
            inserted = False
            for j in range(len(top_points)):
                if fx < top_points[j][1]:
                    top_points.insert(j, (x.copy(), fx))
                    inserted = True
                    break
            if not inserted:
                top_points.append((x.copy(), fx))
            
            # Keep only top 3
            if len(top_points) > 3:
                top_points = top_points[:3]

        # Phase 2: Local Refinement from top 3 points
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Allocate budget to each local search
        # We have 3 points. 
        # Let's give 1/3 of the remaining budget to each, with a minimum.
        # Or just run them sequentially until budget is exhausted.
        
        for x0, f0 in top_points:
            if remaining_budget <= 0:
                break
            
            # Use Nelder-Mead
            # We need to be careful with maxiter.
            # In 5D, Nelder-Mead uses ~6 calls/iter.
            # maxiter = remaining_budget // 6
            # But we share the budget.
            
            # Let's run Nelder-Mead with a wrapper that stops when budget is hit.
            # We can't easily stop scipy mid-iteration.
            # So we estimate maxiter.
            
            # To be safe, we run with a maxiter that fits in the remaining budget.
            # If we run out of calls, the wrapper returns inf, and scipy stops.
            
            calls_left = remaining_budget
            
            def wrapped_func(x):
                nonlocal calls_left
                if calls_left <= 0:
                    return float('inf')
                calls_left -= 1
                return func(x)

            try:
                # Nelder-Mead options
                # xatol and fatol can be tight
                res = so.minimize(wrapped_func, x0, method='Nelder-Mead', 
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                
                # Update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
                remaining_budget = calls_left
            except Exception:
                pass

        return best_x, best_f
