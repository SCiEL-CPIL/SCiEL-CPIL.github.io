"""
Optimizer using Multi-Start Nelder-Mead with Coordinate Polish for BBOB f20 (Schwefel).
Structural change: Combines geometric Nelder-Mead search with 1-D coordinate-wise
line search (Powell's method) to refine solutions in rugged landscapes.
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Track global best
        best_x = np.zeros(n)
        best_fx = float('inf')
        calls = 0
        
        # Multi-start strategy: use restarts to explore different regions
        max_restarts = 10
        restart_budget_min = 40  # minimum calls per restart
        
        def coordinate_polish(x, f, budget_limit):
            """
            Perform coordinate-wise line search (Powell's method style)
            to refine the point x.
            """
            nonlocal calls
            x = x.copy()
            f = f
            # Use a few iterations of coordinate descent
            for _ in range(3): # 3 passes over coordinates
                if calls >= budget_limit or time.time() > deadline:
                    break
                for i in range(n):
                    if calls >= budget_limit or time.time() > deadline:
                        break
                    # Simple ternary search on coordinate i
                    # Range: [-5, 5]
                    a = -5.0
                    b = 5.0
                    # Shrink range based on current x? No, keep global for safety, 
                    # but we can speed up by checking neighbors first.
                    
                    # Try a few steps to find a better point
                    step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
                    best_local_f = f
                    best_local_x = x.copy()
                    
                    for step in step_sizes:
                        x_plus = x.copy()
                        x_plus[i] += step
                        x_minus = x.copy()
                        x_minus[i] -= step
                        
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = np.clip(x_minus, -5, 5)
                        
                        if calls >= budget_limit or time.time() > deadline:
                            break
                        f_plus = func(x_plus)
                        calls += 1
                        if f_plus < best_local_f:
                            best_local_f = f_plus
                            best_local_x = x_plus.copy()
                        
                        if calls >= budget_limit or time.time() > deadline:
                            break
                        f_minus = func(x_minus)
                        calls += 1
                        if f_minus < best_local_f:
                            best_local_f = f_minus
                            best_local_x = x_minus.copy()
                    
                    x = best_local_x
                    f = best_local_f
            
            return x, f

        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Check if we have enough budget for a meaningful restart
            remaining = budget - calls
            if remaining < restart_budget_min:
                break
            
            # For first restart, start at origin; for subsequent, use random start
            if restart == 0:
                x_start = np.zeros(n)
            else:
                # Use a mix of random and perturbed best
                if np.random.rand() < 0.5:
                    x_start = np.random.uniform(-4, 4, size=n)
                else:
                    x_start = best_x + np.random.normal(0, 1.0, size=n)
                    x_start = np.clip(x_start, -4.9, 4.9)
            
            # Initialize Simplex around starting point
            # Use a larger step for initial exploration
            step = 3.0 if restart < 3 else 1.5
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            for i in range(n):
                simplex[i + 1] = x_start
                simplex[i + 1, i] += step
            
            # Clip to box
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            init_ok = True
            for i in range(n + 1):
                if calls >= budget or time.time() > deadline:
                    init_ok = False
                    break
                x = simplex[i]
                fx = func(x)
                fvals[i] = fx
                calls += 1
            
            if not init_ok or calls == 0:
                break
            
            # Update global best from initial evaluation
            local_best_idx = np.argmin(fvals)
            if fvals[local_best_idx] < best_fx:
                best_fx = fvals[local_best_idx]
                best_x = simplex[local_best_idx].copy()
            
            # Run Nelder-Mead for this restart
            # Limit iterations to avoid wasting too much budget on one restart
            max_iter = min(remaining // (n + 2), 25)  # cap iterations
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness (best first)
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Update best
                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()
                
                # Check for convergence: if simplex is very small, restart
                spread = np.max(np.abs(simplex - simplex[0]), axis=0)
                if np.all(spread < 1e-4):
                    break
                
                # Identify worst point
                worst_idx = n
                worst_x = simplex[worst_idx]
                worst_f = fvals[worst_idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:n], axis=0)
                
                # 1. Reflection
                x_ref = centroid + alpha * (centroid - worst_x)
                x_ref = np.clip(x_ref, -5, 5)
                
                if calls >= budget or time.time() > deadline:
                    break
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < fvals[0]:
                    # Reflection is better than best, try Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    
                    if calls >= budget or time.time() > deadline:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        break
                    
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[worst_idx] = x_exp
                        fvals[worst_idx] = f_exp
                    else:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        
                elif f_ref < fvals[n - 1]:
                    # Reflection is better than worst, but worse than second best
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                    
                else:
                    # Reflection is worse than worst, try Contraction
                    x_con = centroid + rho * (worst_x - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < worst_f:
                        simplex[worst_idx] = x_con
                        fvals[worst_idx] = f_con
                    else:
                        # Shrinkage: only shrink towards best
                        for i in range(1, n + 1):
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
            
            # After Nelder-Mead, apply coordinate polish to the best point found
            if calls < budget and time.time() < deadline:
                polish_budget = min(20, budget - calls)
                if polish_budget > 0:
                    x_polished, f_polished = coordinate_polish(simplex[0], fvals[0], budget)
                    if f_polished < best_fx:
                        best_fx = f_polished
                        best_x = x_polished

        return best_x, best_fx
