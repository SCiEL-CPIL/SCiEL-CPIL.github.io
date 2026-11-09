"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Basin Hopping Optimizer for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin: evaluator kills at 330s, task limit is 300s.
        # We aim to finish well before 300s to be safe.
        deadline = time.time() + 280
        
        # Fixed seed for reproducibility in this context, 
        # but using a standard RNG is fine.
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse Exploration (Multi-Start)
        # We will try to sample different regions. 
        # Since dim=5 is small, we can afford a decent number of restarts.
        # Let's allocate ~40% of budget for exploration.
        explore_budget = int(budget * 0.4)
        
        # We want to try about 10-20 distinct starting points.
        # Let's say 15 restarts.
        num_restarts = 15
        budget_per_restart = explore_budget // num_restarts
        
        # If budget_per_restart is too small, reduce restarts
        while budget_per_restart < 5 and num_restarts > 1:
            num_restarts -= 1
            budget_per_restart = explore_budget // num_restarts

        # Store candidates for refinement
        candidates = []

        for r in range(num_restarts):
            # Random start in [-4, 4] to stay within box and likely near optimum
            x = rng.uniform(-4, 4, size=dim)
            
            # Initial evaluation
            if calls >= budget or time.time() > deadline:
                break
                
            fx = func(x)
            calls += 1
            update_best(x, fx)
            
            # Simple local hill climbing (steepest descent with adaptive step)
            # for this specific restart to find a local minimum (which is global for Sphere)
            step_size = 0.5
            local_calls = 0
            
            while local_calls < budget_per_restart:
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate a few perturbations to estimate gradient direction or just do random walk
                # For Sphere, a simple adaptive random walk works well.
                # Let's do a (1+1) style local search with smaller sigma.
                dx = rng.standard_normal(dim)
                x_new = x + step_size * dx
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = func(x_new)
                calls += 1
                local_calls += 1
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    update_best(x, fx)
                    # Increase step size slightly on success
                    step_size *= 1.1
                    step_size = min(step_size, 2.0)
                else:
                    # Decrease step size on failure
                    step_size *= 0.8
                    step_size = max(step_size, 0.01)
                    
            # Save the local minimum found
            candidates.append((fx, x))

        # Phase 2: Refinement of the best candidates
        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Take the top 3 distinct-ish candidates (or just top 3 if few)
        # Actually, since it's Sphere, the best one is likely the global one.
        # We will spend the remaining budget polishing the best one found so far.
        
        # If we have budget left, do a more intensive optimization on the best point.
        # Use a finer-grained search or a more robust local optimizer.
        # Given Sphere is convex, Nelder-Mead or simple coordinate descent is very fast.
        # Let's use a simple adaptive coordinate descent / Nelder-Mead hybrid.
        
        # Let's just continue the (1+1)-ES logic on the best point found,
        # but with a tighter initial sigma and longer horizon.
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.1 # Start small since we are already in a basin
        
        # Remaining budget
        remaining_budget = budget - calls
        
        # Adaptive (1+1)-ES for refinement
        successes = []
        window_size = 10
        
        while calls < budget and time.time() < deadline:
            dx = rng.standard_normal(dim)
            x_new = x + sigma * dx
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            calls += 1
            
            is_success = fx_new < fx
            
            if is_success:
                x = x_new
                fx = fx_new
                update_best(x, fx)
                successes.append(1)
            else:
                successes.append(0)
            
            if len(successes) > window_size:
                successes.pop(0)
            
            if len(successes) == window_size:
                success_rate = sum(successes) / window_size
                if success_rate > 0.3: # Target slightly higher for fine-tuning
                    sigma *= 1.02
                elif success_rate < 0.1:
                    sigma *= 0.95
                else:
                    sigma *= 1.0 # Keep stable
                
                sigma = np.clip(sigma, 1e-8, 5.0)
                
                # If sigma is very small and no progress, break?
                # The loop condition handles budget.

        return best_x, best_f
# EVOLVE-BLOCK-END
