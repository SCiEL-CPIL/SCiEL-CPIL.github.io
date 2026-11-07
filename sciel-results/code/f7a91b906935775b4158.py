"""Optimizer using multi-start Nelder-Mead with adaptive restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best solution found
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def safe_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            return func(x)
        
        def nelder_mead(func, x0, max_iter, initial_alpha=1.0, gamma=1.0, rho=0.5, sigma=2.0):
            """Nelder-Mead simplex method with time guard."""
            nonlocal best_x, best_f, call_count, deadline
            
            n = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                if x0[i] + 0.25 > 5.0:
                    xi[i] = x0[i] - 0.25
                else:
                    xi[i] = x0[i] + 0.25
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if time.time() > deadline:
                    return x0, fvals[0] if fvals else np.inf
                fx = safe_call(simplex[i])
                fvals.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()
            
            fvals = np.array(fvals)
            
            for iteration in range(max_iter):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = fvals[order].copy()
                
                # Compute centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + sigma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
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
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_call(simplex[i])
                
                # Update best
                for i in range(n + 1):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = simplex[i].copy()
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
            
            return simplex[np.argmin(fvals)], np.min(fvals)
        
        # Phase 1: Random exploration with Nelder-Mead refinement
        # Start with a few random points, then do Nelder-Mead from each
        n_initial = min(5, self.budget // 50)
        
        for i in range(n_initial):
            if time.time() > deadline or call_count >= self.budget:
                break
            
            # Random initial point
            x0 = rng.uniform(-5, 5, self.dim)
            fx = safe_call(x0)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            # Nelder-Mead from this point
            remaining_budget = self.budget - call_count
            max_iter = min(200, remaining_budget // (self.dim + 1))
            if max_iter < 10:
                continue
            
            x_best_local, f_best_local = nelder_mead(func, x0, max_iter)
        
        # Phase 2: More aggressive search if budget remains
        if time.time() < deadline - 5 and call_count < self.budget - 10:
            # Try different scales
            for scale in [3.0, 2.0, 1.0, 0.5]:
                if time.time() > deadline or call_count >= self.budget - 10:
                    break
                
                # Random point in smaller box
                x0 = rng.uniform(-scale, scale, self.dim)
                fx = safe_call(x0)
                if fx < best_f:
                    best_f = fx
                    best_x = x0.copy()
                
                remaining_budget = self.budget - call_count
                max_iter = min(100, remaining_budget // (self.dim + 1))
                if max_iter < 10:
                    continue
                
                x_best_local, f_best_local = nelder_mead(func, x0, max_iter)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
