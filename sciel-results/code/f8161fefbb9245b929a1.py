"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive initial simplex sizes for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Strategy: Multi-start Nelder-Mead
        # 1. Generate a set of diverse starting points
        # 2. Run a limited-budget Nelder-Mead from each start
        # 3. Keep the global best

        # We will use a portion of the budget for initial exploration (finding good basins)
        # and the rest for local refinement.
        # However, NM is fast in convergence but slow in exploration.
        # Let's allocate ~30% of budget to a coarse grid/random search to find good basins,
        # then use the remaining 70% for NM refinement from the best few points found.
        
        # Phase 1: Coarse Exploration
        # Use random points and maybe a few deterministic ones (center, corners)
        exploration_budget = int(budget * 0.3)
        
        candidates = []
        
        # Deterministic points: center
        center = np.zeros(dim)
        f_center = func(center)
        calls_made += 1
        update_best(center, f_center)
        candidates.append((center, f_center))
        
        # Random points
        while calls_made < exploration_budget and not check_time():
            x = np.random.uniform(-4, 4, dim)  # Minimizer is in [-4, 4]
            f = func(x)
            calls_made += 1
            update_best(x, f)
            candidates.append((x, f))
        
        if best_x is None:
            return best_x, best_f

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[1])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # We will pick the top K candidates and run NM from each.
        # How many? Maybe top 5?
        # Budget remaining: budget - calls_made
        # Each NM run should get a fair share.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Pick top 5 unique-ish points (avoid duplicates)
        # For simplicity, just take the top 5 from the sorted list.
        top_k = min(5, len(candidates))
        starts = [candidates[i][0] for i in range(top_k)]
        
        # Allocate budget for each NM run
        # Reserve some budget for potential failures or overhead
        nm_budget_per_run = max(10, int(remaining_budget / top_k))
        
        for start_point in starts:
            if calls_made >= budget or check_time():
                break
            
            # Create initial simplex for NM
            # We need dim+1 points.
            # Start with start_point, and perturb each dimension.
            # Perturbation size: adaptive based on previous failures?
            # Let's use a small perturbation, e.g., 0.1 or 0.5.
            # If NM fails to improve, we can try a larger perturbation.
            
            # We'll implement a simple NM loop manually to control the budget precisely.
            # Or use scipy? The previous program used scipy.
            # Let's implement a lightweight NM to avoid overhead and have full control.
            
            def run_nm(start_x, max_calls):
                nonlocal calls_made, best_x, best_f
                
                if max_calls < dim + 1:
                    return
                
                # Initialize simplex
                # Simplex: start_x, start_x + delta*e_i
                delta = 0.5
                simplex = [start_x.copy()]
                for i in range(dim):
                    p = start_x.copy()
                    p[i] += delta
                    p = np.clip(p, -5, 5)
                    simplex.append(p)
                
                # Evaluate simplex
                f_vals = []
                for p in simplex:
                    if calls_made >= budget or check_time():
                        return
                    f_p = func(p)
                    calls_made += 1
                    f_vals.append(f_p)
                    update_best(p, f_p)
                
                # NM Loop
                # Parameters
                alpha = 1.0  # Reflection
                gamma = 2.0  # Expansion
                rho = 0.5    # Contraction
                sigma = 0.5  # Shrink
                max_iter = max_calls // (dim + 1)  # Rough estimate
                
                for _ in range(max_iter):
                    if calls_made >= budget or check_time():
                        break
                    
                    # Sort simplex by f_vals
                    order = np.argsort(f_vals)
                    simplex = [simplex[i] for i in order]
                    f_vals = [f_vals[i] for i in order]
                    
                    # Centroid of all points except the worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Worst point
                    x_worst = simplex[-1]
                    f_worst = f_vals[-1]
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    calls_made += 1
                    update_best(x_r, f_r)
                    
                    if f_r < f_vals[0]:  # Better than best
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls_made += 1
                        update_best(x_e, f_e)
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_vals[-1] = f_r
                    elif f_r < f_vals[-2]:  # Better than second worst
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    else:
                        # Contraction
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls_made += 1
                        update_best(x_c, f_c)
                        
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Inside contraction
                            x_ic = centroid - rho * (x_r - centroid)
                            x_ic = np.clip(x_ic, -5, 5)
                            f_ic = func(x_ic)
                            calls_made += 1
                            update_best(x_ic, f_ic)
                            
                            if f_ic <= f_worst:
                                simplex[-1] = x_ic
                                f_vals[-1] = f_ic
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], -5, 5)
                                    if calls_made >= budget or check_time():
                                        break
                                    f_shrink = func(simplex[i])
                                    calls_made += 1
                                    f_vals[i] = f_shrink
                                    update_best(simplex[i], f_shrink)
                                break # Break after shrink as it's a drastic change
                
            run_nm(start_point, nm_budget_per_run)
            
            # If we still have budget, we could restart NM from the new best if it improved significantly
            # But for now, let's just move to the next start point.
            # Actually, if the best point has changed significantly, it might be worth restarting NM from the new best.
            # Let's add a simple check: if best_f has improved by a lot, restart NM from best_x with remaining budget.
            # But this might be complex. Let's stick to the multi-start for now.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
