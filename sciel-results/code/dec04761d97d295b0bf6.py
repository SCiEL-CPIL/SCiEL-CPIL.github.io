"""
Nelder-Mead optimizer with plateau-breaking for BBOB f7 (Step ellipsoidal) in 5D.

Strategy:
1. Nelder-Mead is a derivative-free local optimizer that works well on smooth-ish landscapes.
2. BBOB f7 has "steps" (flat regions) which can stall gradient-based or even NM methods if they land exactly on a plateau.
3. We implement a custom Nelder-Mead loop that detects stagnation (all simplex points having nearly identical fitness) and perturbs the worst point with a random kick to escape the plateau.
4. We use a multi-start approach: a few initial random starts to increase the chance of landing in the basin of the global minimum, then refine the best one.
5. We strictly track the number of function evaluations to stay within the budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Nelder-Mead with plateau-breaking.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                # Return a dummy value to avoid errors, but we shouldn't be here
                return 0.0 
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def nelder_mead(func, x0, budget_remaining, max_iter=1000):
            """
            Custom Nelder-Mead implementation with plateau handling.
            Returns best_x, best_f, calls_used.
            """
            n = len(x0)
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0
                simplex[i, i-1] += 0.5 if x0[i-1] <= 0 else -0.5 # Step size 0.5 is reasonable for [-5,5]
            
            # Evaluate initial simplex
            fvals = np.array([func(x) for x in simplex])
            calls_used = n + 1
            
            # Parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage
            
            # Stagnation counter
            stagnation_count = 0
            prev_best_f = np.inf
            
            for i_iter in range(max_iter):
                if calls_used >= budget_remaining:
                    break
                if time.time() > deadline:
                    break
                    
                # Sort simplex by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                best_f_val = fvals[0]
                worst_f_val = fvals[-1]
                second_worst_f_val = fvals[-2]
                
                # Check convergence/stagnation
                # If the range of fitness values is very small, we might be stuck on a plateau or converged
                if np.abs(worst_f_val - best_f_val) < 1e-10:
                    # Try to break the plateau by perturbing the worst point
                    # If we've already tried this a few times, break
                    if stagnation_count > 5:
                        break
                    stagnation_count += 1
                    # Perturb the worst point
                    noise = np.random.randn(n) * 0.5
                    new_worst = simplex[-1] + noise
                    new_worst = np.clip(new_worst, -5.0, 5.0)
                    new_f = func(new_worst)
                    calls_used += 1
                    if new_f < worst_f_val:
                        simplex[-1] = new_worst
                        fvals[-1] = new_f
                    else:
                        # If perturbation didn't help, shrink the simplex around the best point
                        # This is a fallback for true convergence
                        if i_iter > 50: 
                             break
                        # Shrink
                        centroid = simplex[:-1].mean(axis=0)
                        for j in range(1, n + 1):
                            simplex[j] = centroid + sigma * (simplex[j] - centroid)
                            simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                            fvals[j] = func(simplex[j])
                            calls_used += 1
                        continue
                else:
                    stagnation_count = 0
                
                # Calculate centroid of all points except the worst
                centroid = simplex[:-1].mean(axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                calls_used += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < second_worst_f_val:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if fr < worst, inside if fr > worst
                    if fr < worst_f_val:
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = func(xc)
                        calls_used += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, n + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                                fvals[j] = func(simplex[j])
                                calls_used += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fc = func(xc)
                        calls_used += 1
                        if fc < worst_f_val:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, n + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                                fvals[j] = func(simplex[j])
                                calls_used += 1
                            
                # Check for tiny simplex (convergence)
                simplex_spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.all(simplex_spread < 1e-6):
                    break
                    
            # Find best in final simplex
            final_best_idx = np.argmin(fvals)
            return simplex[final_best_idx], fvals[final_best_idx], calls_used

        # Multi-start strategy
        # 1. Start at center
        # 2. Start at a few random points
        # We have 500 calls. NM uses ~10-20 calls per iteration.
        # Let's do 1 center start and 4 random starts.
        # Or better: do a quick random search to find a good starting point, then NM.
        
        # Phase 1: Random Search to find a good basin
        # Use 50% of budget for exploration? No, NM is fast.
        # Let's use 50 calls for random search to pick the best starting point.
        num_random_starts = 50
        if budget > num_random_starts + 10:
            best_start_x = None
            best_start_f = np.inf
            for i in range(num_random_starts):
                if calls_made >= budget:
                    break
                x_rand = np.random.uniform(-4, 4, dim) # Min is in [-4,4]
                f_rand = safe_func(x_rand)
                if f_rand < best_start_f:
                    best_start_f = f_rand
                    best_start_x = x_rand
            
            # Phase 2: Nelder-Mead refinement on the best start
            if best_start_x is not None:
                remaining_budget = budget - calls_made
                # NM needs at least n+1 calls. 
                # We pass the safe_func which tracks calls.
                # But we need to pass the remaining budget to NM to stop it.
                # The safe_func will stop if calls_made >= budget.
                # We can just let NM run until it stops or budget is hit.
                
                # To be safe, we can limit iterations in NM based on remaining budget.
                # Each NM iteration uses ~1-4 calls.
                max_nm_iter = max(10, remaining_budget // 4)
                
                nm_x, nm_f, nm_calls = nelder_mead(safe_func, best_start_x, remaining_budget, max_iter=max_nm_iter)
                
                if nm_f < best_f:
                    best_x = nm_x
                    best_f = nm_f

        # If budget was very small or something failed, ensure we have a best_x
        if best_x is None:
             # Fallback: return the center if no calls were made or all failed
             best_x = np.zeros(dim)
             if calls_made < budget:
                 best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
