"""Optimizer for BBOB f23 (Katsuura) using Multi-Start (1+1)-ES with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with LHS initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # We use 20 points for initial spread to leave more budget for local search
        n_lhs = 20
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        # Phase 2: Multi-Start Local Search
        # We divide the remaining budget into trials.
        # Each trial consists of a "Jump" from the current best (or random) 
        # followed by a local (1+1)-ES refinement.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Reserve budget for final polishing phase
        reserve_polish = min(30, remaining_budget // 5)
        search_budget = remaining_budget - reserve_polish
        
        # Define trial structure
        # We want roughly 50-60 steps per trial to allow deeper convergence
        trial_length = 55
        n_trials = search_budget // trial_length
        
        # If we can't do a full trial, do one big remaining chunk
        if n_trials == 0:
            trial_length = search_budget
            n_trials = 1

        sigma_init = 1.0
        min_sigma = 0.0005  # Lower min_sigma for finer convergence
        max_sigma = 3.0
        jump_scale = 2.0 # Scale of jump from best to escape basin

        for t in range(n_trials):
            if calls_made >= budget - reserve_polish:
                break
            
            # Determine starting point for this trial
            # Strategy: Jump from global best with some randomness
            # If this is the first trial after LHS, we can also try a pure random point 
            # to ensure we don't just refine the LHS best too much if it's bad.
            if t == 0:
                # First trial: Start from the best LHS point, but maybe jump a bit?
                # Or just refine it. Let's refine it first.
                x = x_global_best.copy()
            else:
                # Subsequent trials: Jump from global best to explore new basin
                # The jump should be large enough to escape the current local min
                # but not so large it goes out of bounds (handled by clip)
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            
            # Local (1+1)-ES Phase
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            success_count = 0
            fail_count = 0
            window_size = 5
            
            # Run local search for up to trial_length steps
            for step in range(trial_length - 1): # -1 because we already evaluated x
                if calls_made >= budget - reserve_polish:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    fail_count = 0
                    update_best(x, fx)
                else:
                    fail_count += 1
                    success_count = 0
                
                # Adapt sigma based on recent successes/failures
                if success_count >= window_size:
                    sigma = min(sigma * 1.3, max_sigma)
                    success_count = 0
                elif fail_count >= window_size:
                    sigma = max(sigma * 0.7, min_sigma)
                    fail_count = 0
                    
        # Phase 3: Final fine-tuning with small steps
        # Use the remaining budget for very fine local search around the best point
        remaining = budget - calls_made
        if remaining > 5 and x_global_best is not None:
            x = x_global_best.copy()
            fx = fx_global_best
            sigma = 0.05
            
            for _ in range(remaining - 2):  # Keep 2 calls as safety buffer
                if calls_made >= budget - 2:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                # Generate candidate with small step
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.1, 0.2)  # Slightly increase on success
                else:
                    sigma = max(sigma * 0.8, 0.001)  # Decrease on failure
                    
        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
