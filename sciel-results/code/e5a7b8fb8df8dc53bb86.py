"""
Optimization strategy:
1. Initial random sampling to identify promising basins.
2. Multi-restart adaptive (1+1)-ES for local refinement.
   - Uses a success-rate based step size adaptation for robustness.
   - Restarts are budgeted to allow exploration of multiple basins.
3. Final high-precision polish around the global best.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s, we stop at 280s to be safe
        # and ensure we return before any hard timeouts.
        deadline = time.time() + 280.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Exploratory Random Search
        # Sample points to find good starting basins.
        # Rastrigin has many local minima. We need a decent spread.
        n_explore = min(40, budget // 8)
        
        # Generate random points
        explore_x = rng.uniform(-5, 5, (n_explore, dim))
        explore_f = np.empty(n_explore)
        
        for i in range(n_explore):
            if check_deadline():
                break
            explore_f[i] = evaluate(explore_x[i])
        
        # If we broke early, adjust n_explore
        if calls_made < n_explore:
            n_explore = calls_made
            explore_f = explore_f[:n_explore]

        # Sort by fitness and select top seeds
        # We want diverse good points, not just the single best.
        sort_idx = np.argsort(explore_f)
        
        # Select up to 5 seeds. 
        # We prioritize the best, but if we have budget, we can try others.
        # Let's take the top 5 unique-ish points.
        seeds = []
        for idx in sort_idx:
            if len(seeds) >= 5:
                break
            # Simple diversity check: distance from existing seeds
            # For D=5, we can just take the top 5 if they are distinct enough
            # Or just take the top 5. Given the budget, 5 restarts is reasonable.
            seeds.append(explore_x[idx].copy())

        # Phase 2: Local Refinement with Adaptive (1+1)-ES
        # We allocate the remaining budget to these seeds.
        # Simple strategy: give each seed a fixed number of iterations,
        # or run until time/budget is low.
        
        # Let's estimate iterations per seed.
        # Total remaining budget: budget - calls_made
        # We have up to 5 seeds.
        # Let's reserve 20% of remaining budget for final polish.
        remaining_budget = budget - calls_made
        polish_budget = max(20, int(remaining_budget * 0.2))
        refine_budget = remaining_budget - polish_budget
        
        if refine_budget <= 0:
            # Not enough budget for refinement, skip to polish
            pass
        else:
            # Distribute refine_budget among seeds
            # More seeds if budget allows, but 5 is a good cap.
            n_seeds = len(seeds)
            if n_seeds == 0:
                # Fallback if exploration failed
                seed = rng.uniform(-5, 5, dim)
                seeds = [seed]
                n_seeds = 1
            
            iters_per_seed = max(1, refine_budget // n_seeds)
            
            for seed in seeds:
                if check_deadline():
                    break
                    
                x = seed.copy()
                # Re-evaluate seed? We already have the value from Phase 1.
                # But we need to track local best for this run.
                # If this seed is the global best, fx is best_f.
                # Otherwise, we should evaluate it to be sure, or use the cached value.
                # Caching is better to save calls.
                # Find index of this seed in explore_x?
                # Easier: just evaluate it if it's not the global best.
                # Actually, let's just start the ES from the seed.
                # We need the current fx.
                if np.allclose(x, best_x):
                    fx = best_f
                else:
                    # We have the value from explore_f, but let's be safe and 
                    # just use the value we computed. 
                    # Wait, we don't have a direct mapping from seed to explore_f index 
                    # in this loop structure easily. 
                    # Let's just evaluate it. It's 1 call.
                    # But wait, we already evaluated it in Phase 1.
                    # Let's store the values with the seeds.
                    # Re-doing seed selection to keep (x, f) pairs.
                    pass
                
            # Let's re-structure Phase 2 to keep (x, f) pairs to avoid re-evaluation.
            # Clear previous loop logic.
            
            # Re-select seeds with values
            seeds_with_val = []
            for idx in sort_idx:
                if len(seeds_with_val) >= 5:
                    break
                seeds_with_val.append((explore_x[idx].copy(), explore_f[idx]))
                
            if not seeds_with_val:
                # Fallback
                x0 = rng.uniform(-5, 5, dim)
                f0 = evaluate(x0)
                seeds_with_val = [(x0, f0)]
            
            n_seeds = len(seeds_with_val)
            iters_per_seed = max(1, refine_budget // n_seeds)
            
            for x_init, f_init in seeds_with_val:
                if check_deadline():
                    break
                
                x = x_init.copy()
                fx = f_init
                sigma = 1.0  # Start with reasonable step size
                success_count = 0
                total_steps = 0
                # Target success rate ~ 1/5 = 0.2
                # If success rate > 0.2, increase sigma. Else decrease.
                # We'll adjust sigma every 10 steps or so.
                
                for _ in range(iters_per_seed):
                    if check_deadline():
                        break
                    
                    # Generate perturbation
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    total_steps += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        success_count += 1
                    else:
                        pass
                    
                    # Adapt sigma every 10 steps
                    if total_steps % 10 == 0:
                        success_rate = success_count / 10.0
                        if success_rate > 0.2:
                            sigma *= 1.5
                        else:
                            sigma *= 0.75
                        success_count = 0
                        
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 3: Final Polish
        # Use remaining budget for a very fine-grained search around global best.
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            # Start with small sigma
            sigma = 0.1
            success_count = 0
            total_steps = 0
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                
                # Adapt sigma more aggressively for polish
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    success_count = 0
                    
                sigma = np.clip(sigma, 1e-8, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
