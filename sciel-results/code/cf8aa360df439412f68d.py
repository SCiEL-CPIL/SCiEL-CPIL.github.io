"""Optimizer for BBOB-style f17 (Schaffers F7) using Multistart Nelder-Mead."""
import numpy as np
import time


class Optimizer:
    """Multistart Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        # Track global best
        global_best_x = None
        global_best_f = float('inf')
        
        # State for Nelder-Mead
        # We will run multiple restarts if stagnation occurs or budget allows
        # Nelder-Mead uses a simplex of dim+1 points.
        # Each "step" in NM involves 1 function evaluation (reflection, expansion, etc.)
        # Budget 500 allows for ~400-450 NM steps if we reserve some for initial exploration.
        
        # Strategy:
        # 1. Start with a reasonable initial simplex.
        # 2. Run NM until convergence or stagnation.
        # 3. If stagnated, restart from a new random point or a perturbed best point.
        # 4. Keep the best solution found across all runs.
        
        # To maximize AOCC, we want fast initial progress.
        # NM is very fast locally. 
        # Schaffers F7 has local minima. We need to escape them.
        
        # Let's allocate budget:
        # 10% for initial coarse grid/random sampling to find a good basin?
        # Or just jump straight into NM from center?
        # Center (0,0,0,0,0) is often a good start for BBOB instances shifted within [-4,4].
        
        # Let's try a hybrid:
        # Phase 1: 50 evaluations of random points to identify a promising region.
        # Phase 2: Nelder-Mead starting from the best of Phase 1.
        # Phase 3: If NM stagnates, restart NM from other good points from Phase 1 or new randoms.
        
        # Actually, NM from center is usually very fast.
        # Let's do:
        # 1. Evaluate center.
        # 2. Run NM from center.
        # 3. If NM stagnates (no improvement in 20 evals), restart from a random point near best.
        
        # Simpler and more robust:
        # Run NM. If it converges, it's done. If not, it's stuck.
        # Since we have 500 calls, and NM takes ~dim*10 calls to converge locally,
        # we can afford several restarts.
        
        # Let's implement a "Restarting Nelder-Mead"
        
        calls_made = 0
        max_calls = budget - 5 # Reserve 5 for safety
        
        def nelder_mead(func, x0, budget_remaining, rng, current_best_f, current_best_x):
            nonlocal calls_made
            nonlocal global_best_f
            nonlocal global_best_x
            
            # Initialize simplex
            # Standard initialization: x0, x0 + delta*e_i
            # Choose delta based on box size
            delta = 0.5
            
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate initial simplex
            simplex_f = []
            for i in range(dim + 1):
                if calls_made >= budget_remaining:
                    return current_best_f, current_best_x
                simplex_f.append(func(simplex[i]))
                calls_made += 1
                if simplex_f[-1] < current_best_f:
                    current_best_f = simplex_f[-1]
                    current_best_x = simplex[i].copy()
            
            # Sort simplex
            def sort_simplex():
                nonlocal simplex, simplex_f
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
            
            sort_simplex()
            
            # NM Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            stagnation_count = 0
            best_f_so_far_in_run = simplex_f[0]
            best_x_so_far_in_run = simplex[0].copy()
            
            while calls_made < budget_remaining:
                # Check for convergence
                if np.std(simplex_f) < 1e-8:
                    break
                
                # Reflection
                centroid = np.sum(simplex[:-1], axis=0) / dim
                xr = centroid + alpha * (centroid - simplex[-1])
                
                fr = func(xr)
                calls_made += 1
                if fr < current_best_f:
                    current_best_f = fr
                    current_best_x = xr.copy()
                
                if fr < simplex_f[0]:
                    # Try Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(xe)
                    calls_made += 1
                    if fe < current_best_f:
                        current_best_f = fe
                        current_best_x = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept Reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = func(xc)
                        calls_made += 1
                        if fc < current_best_f:
                            current_best_f = fc
                            current_best_x = xc.copy()
                        
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(simplex[i])
                                calls_made += 1
                                if simplex_f[i] < current_best_f:
                                    current_best_f = simplex_f[i]
                                    current_best_x = simplex[i].copy()
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - xr)
                        fc = func(xc)
                        calls_made += 1
                        if fc < current_best_f:
                            current_best_f = fc
                            current_best_x = xc.copy()
                        
                        if fc <= simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = func(simplex[i])
                                calls_made += 1
                                if simplex_f[i] < current_best_f:
                                    current_best_f = simplex_f[i]
                                    current_best_x = simplex[i].copy()
                
                sort_simplex()
                
                # Stagnation check
                if simplex_f[0] < best_f_so_far_in_run - 1e-6:
                    best_f_so_far_in_run = simplex_f[0]
                    best_x_so_far_in_run = simplex[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                # If stuck for too long, break to allow restart
                if stagnation_count > 20:
                    break
                
                # Time check
                if time.time() > deadline - 5.0:
                    return current_best_f, current_best_x
            
            return current_best_f, current_best_x

        # Initial Point: Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        global_best_f = f0
        global_best_x = x0.copy()
        
        # Run 1: From Center
        if calls_made < max_calls:
            f_best, x_best = nelder_mead(func, x0, max_calls, rng, global_best_f, global_best_x)
            global_best_f = f_best
            global_best_x = x_best
        
        # If we still have budget and haven't found a very good solution, restart
        # Schaffers F7 minimum is often not at 0.
        # Let's try a few random restarts if we have time/budget.
        
        # Estimate remaining budget
        # NM typically consumes ~dim * 15 calls per local optimum.
        # If we used < 200 calls, we can restart.
        
        if calls_made < max_calls - 10:
            # Try 1 restart from a random point
            x_start = rng.uniform(-3, 3, dim)
            f_start = func(x_start)
            calls_made += 1
            if f_start < global_best_f:
                global_best_f = f_start
                global_best_x = x_start.copy()
            
            if calls_made < max_calls:
                f_best, x_best = nelder_mead(func, x_start, max_calls, rng, global_best_f, global_best_x)
                global_best_f = f_best
                global_best_x = x_best
        
        # If still budget, try another restart
        if calls_made < max_calls - 10:
            x_start = rng.uniform(-4, 4, dim)
            f_start = func(x_start)
            calls_made += 1
            if f_start < global_best_f:
                global_best_f = f_start
                global_best_x = x_start.copy()
            
            if calls_made < max_calls:
                f_best, x_best = nelder_mead(func, x_start, max_calls, rng, global_best_f, global_best_x)
                global_best_f = f_best
                global_best_x = x_best
        
        # Final check
        if time.time() > deadline - 2.0:
            return global_best_x, global_best_f

        # If we have significant budget left and the solution is not "perfect" (hard to know),
        # we can do a final fine-tuning with a smaller delta NM or (1+1) ES.
        # But NM should have converged.
        
        return global_best_x, global_best_f
