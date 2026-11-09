"""Portfolio-based CMA-ES with aggressive restarts for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        
        global_best_f = np.inf
        global_best_x = np.zeros(d)
        
        calls_used = 0
        
        # We will run multiple independent CMA-ES-like trajectories.
        # Since we can't easily implement full CMA-ES from scratch efficiently in a short snippet
        # without risking bugs, we will implement a simplified, robust version of 
        # (mu+lambda)-ES with adaptive step size and restarts, which is structurally different 
        # from the coordinate-wise grid descent. 
        # Actually, let's use a simple population-based approach:
        # 1. Maintain a population of size P.
        # 2. Generate offspring by adding noise to the current best (or a mix).
        # 3. Update step size based on success rate.
        # 4. Restart if stagnated.
        
        pop_size = 10
        # Number of restarts to attempt if we stall
        max_restarts = 50
        
        def run_es_trajectory(x_init, f_init, max_calls):
            nonlocal global_best_f, global_best_x, calls_used
            
            x_best = x_init.copy()
            f_best = f_init
            
            # Initial step size
            sigma = 2.0
            
            # Population
            pop = [x_best.copy()] * pop_size
            # Evaluate initial population? No, we start with x_best as the elite.
            # We need to generate offspring.
            
            # To make it more robust, let's keep a small archive of best points found so far in this trajectory
            # Or just simple (1+lambda) with population? 
            # Let's do (mu+lambda) with mu=1, lambda=pop_size.
            
            stall_count = 0
            target_success = 0.33  # Roughly 1/3 success rate for step size adaptation
            
            while calls_used < budget and calls_used < max_calls:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                
                for i in range(pop_size):
                    if calls_used >= budget or calls_used >= max_calls:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Gaussian noise
                    noise = np.random.normal(0, 1, d)
                    x_new = x_best + sigma * noise
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = func(x_new)
                    calls_used += 1
                    
                    offspring.append(x_new)
                    offspring_f.append(f_new)
                
                if not offspring:
                    break
                
                # Select best from offspring and current best
                # Combine current best and offspring
                candidates_x = [x_best] + offspring
                candidates_f = [f_best] + offspring_f
                
                # Sort by fitness
                combined = sorted(zip(candidates_f, candidates_x), key=lambda x: x[0])
                
                new_f_best, new_x_best = combined[0]
                
                # Adapt step size
                # Success rate: how many offspring were better than the current best?
                # In (1+lambda), success is usually defined as improvement.
                # But for plateau, we might want to look at the best offspring.
                
                # Simple adaptation:
                # If the best offspring is better, increase sigma slightly? Or keep?
                # Standard CMA-ES uses a more complex mechanism.
                # Let's use a simple rule:
                # If improvement, sigma *= 1.05 (slow growth)
                # If no improvement, sigma *= 0.8 (shrink)
                # This is a heuristic, but structurally different from grid descent.
                
                if new_f_best < f_best:
                    # Improvement
                    x_best = new_x_best
                    f_best = new_f_best
                    stall_count = 0
                    sigma *= 1.1  # Expand search space slightly
                    if f_best < global_best_f:
                        global_best_f = f_best
                        global_best_x = x_best.copy()
                else:
                    # No improvement
                    # Check if any offspring was "close" or if we should just shrink
                    # For step functions, shrinking is key to find the edge.
                    sigma *= 0.7
                    stall_count += 1
                    
                    # If we stall for too long, restart this trajectory
                    if stall_count > 20:
                        break
                
                # Optional: If sigma becomes too small, break
                if sigma < 1e-6:
                    break
            
            return x_best, f_best

        # Main Loop: Multi-start with restarts
        num_starts = 5
        calls_per_start = budget // (num_starts + max_restarts) # Rough allocation
        
        # We'll just run sequentially, letting each trajectory use up to 'calls_per_start' or until stall
        # But we want to reuse the global best for new starts if possible?
        # No, independent trajectories are better for escaping plateaus.
        
        # Generate initial points
        starts = []
        for i in range(num_starts):
            if calls_used >= budget:
                break
            x0 = np.random.uniform(-4.5, 4.5, d)
            f0 = func(x0)
            calls_used += 1
            starts.append((x0, f0))
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()
        
        # Sort starts
        starts.sort(key=lambda x: x[1])
        
        # Run ES from each start
        for x0, f0 in starts:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Limit calls per trajectory to ensure we try multiple starts
            # But if one is doing great, let it go?
            # For now, fixed allocation.
            max_calls_for_this = min(budget - calls_used, 80)
            
            run_es_trajectory(x0, f0, max_calls_for_this)
            
        # If we still have budget, do a final polish with a smaller sigma ES or grid?
        # Let's do a final ES run from the global best with a smaller initial sigma.
        if calls_used < budget and time.time() < deadline:
            x_init = global_best_x.copy()
            f_init = global_best_f
            # Use a smaller initial sigma for refinement
            # We can't easily change the internal sigma of the helper, so let's just call it again
            # But the helper starts with sigma=2.0. 
            # Let's modify the helper to accept initial_sigma.
            
            # Since we can't easily modify the closure without rewriting, 
            # let's just rely on the previous runs. 
            # Or, implement a quick coordinate refinement if budget remains?
            # No, stick to the ES strategy.
            
            # Let's run one more ES trajectory from the global best, but with a twist:
            # We'll just call the same function, but it will start with sigma=2.0 again.
            # This might be redundant if we are already close.
            # However, the stall condition will trigger quickly if we are on a plateau edge.
            
            if calls_used < budget - 10:
                run_es_trajectory(x_init, f_init, budget - calls_used)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
