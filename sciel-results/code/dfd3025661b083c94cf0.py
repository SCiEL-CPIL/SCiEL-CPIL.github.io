"""
Optimizer using Multi-Start Nelder-Mead with Adaptive Coordinate Polish for BBOB f20 (Schwefel).
Structural refinement: Tightens coordinate polish with golden-section search, adds
adaptive step sizing in NM, and increases polish frequency to improve early-progress AOCC.
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
        max_restarts = 12
        restart_budget_min = 30  # minimum calls per restart
        
        def golden_section_1d(x, i, f, a, b, max_iter=10):
            """
            Golden section search along dimension i of x.
            Returns (new_x, new_f) after searching in [a, b].
            """
            nonlocal calls
            invphi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
            invphi2 = invphi * invphi
            
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            x_c = x.copy()
            x_c[i] = c
            x_c = np.clip(x_c, -5, 5)
            if calls >= budget or time.time() > deadline:
                return x, f
            f_c = func(x_c)
            calls += 1
            
            x_d = x.copy()
            x_d[i] = d
            x_d = np.clip(x_d, -5, 5)
            if calls >= budget or time.time() > deadline:
                return x, f
            f_d = func(x_d)
            calls += 1
            
            best_x = x.copy()
            best_f = f
            if f_c < best_f:
                best_f = f_c
                best_x = x_c.copy()
            if f_d < best_f:
                best_f = f_d
                best_x = x_d.copy()
            
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - invphi * (b - a)
                    x_c = x.copy()
                    x_c[i] = c
                    x_c = np.clip(x_c, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + invphi * (b - a)
                    x_d = x.copy()
                    x_d[i] = d
                    x_d = np.clip(x_d, -5, 5)
                    if calls >= budget or time.time() > deadline:
                        break
                    f_d = func(x_d)
                    calls += 1
                    if f_d < best_f:
                        best_f = f_d
                        best_x = x_d.copy()
            
            return best_x, best_f
        
        def coordinate_polish(x, f, budget_limit):
            """
            Perform coordinate-wise golden section search (Powell's method style)
            to refine the point x.
            """
            nonlocal calls
            x = x.copy()
            f = f
            # Use 2 passes over coordinates with tighter ranges on second pass
            for pass_num in range(2):
                if calls >= budget_limit or time.time() > deadline:
                    break
                for i in range(n):
                    if calls >= budget_limit or time.time() > deadline:
                        break
                    # Search range: narrow around current x
                    if pass_num == 0:
                        half_width = 2.0
                    else:
                        half_width = 0.5
                    a = max(-5.0, x[i] - half_width)
                    b = min(5.0, x[i] + half_width)
                    
                    max_iter = 8 if pass_num == 0 else 6
                    x, f = golden_section_1d(x, i, f, a, b, max_iter)
                    if f < best_fx:
                        best_fx = f
                        best_x = x.copy()
            
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
            # Use adaptive step: larger for early restarts, smaller later
            step = 3.0 if restart < 2 else (2.0 if restart < 5 else 1.0)
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
            # More iterations for early restarts when budget is plentiful
            max_iter = min(remaining // (n + 2), 30)  # cap iterations
            
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
                if np.all(spread < 1e-5):
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
                polish_budget = min(30, budget - calls)
                if polish_budget > 0:
                    x_polished, f_polished = coordinate_polish(simplex[0], fvals[0], budget)
                    if f_polished < best_fx:
                        best_fx = f_polished
                        best_x = x_polished

        # Final polish pass with remaining budget
        if calls < budget and time.time() < deadline:
            final_budget = budget - calls
            if final_budget > 10:
                x_final, f_final = coordinate_polish(best_x, best_fx, budget)
                if f_final < best_fx:
                    best_fx = f_final
                    best_x = x_final
        
        return best_x, best_fx
