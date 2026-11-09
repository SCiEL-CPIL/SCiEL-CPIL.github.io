# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive (1+1)-ES with coordinate descent warm start, 1/5 rule, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Coordinate Descent Warm Start
        # This gives early progress which is crucial for AOCC scoring.
        # We perform a simple coordinate descent from the center.
        
        # Start at center
        x = np.zeros(n)
        fx = wrapped_func(x)
        
        # Initial step size for coordinate descent
        cd_step = 2.0
        min_cd_step = 0.05
        
        # Perform a few rounds of coordinate descent
        # Each round tries to improve one coordinate at a time
        for _ in range(5):  # 5 rounds
            if calls_made[0] >= budget or time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    trial = x.copy()
                    trial[i] = np.clip(trial[i] + sign * cd_step, -5, 5)
                    f_trial = wrapped_func(trial)
                    if f_trial < fx:
                        x = trial
                        fx = f_trial
                        improved = True
            if not improved:
                # If no improvement, shrink step size
                cd_step *= 0.5
                if cd_step < min_cd_step:
                    break

        # Phase 2: Adaptive (1+1)-ES with restarts
        # Now refine using (1+1)-ES starting from the best point found by CD.
        
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 30  # Steps without improvement before restarting
        max_sigma = 5.0       # Cap on sigma to prevent divergence
        min_sigma = 1e-5      # Floor on sigma
        
        # Initialize ES state
        x_es = x.copy()
        fx_es = fx
        sigma = 0.5  # Start with smaller sigma since we are already refined
        
        steps_without_improvement = 0

        while calls_made[0] < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x_es + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx_es:
                x_es, fx_es = y, fy
                successes += 1
                steps_without_improvement = 0
                # Increase sigma on success, but cap it
                sigma = min(sigma * 1.15, max_sigma)
            else:
                steps_without_improvement += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.9, min_sigma)

            steps_in_window += 1
            
            # Adapt step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.05, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)
                successes = 0
                steps_in_window = 0

            # Stall detection: if no improvement for a while, restart
            if steps_without_improvement > stall_threshold:
                # Reset local variables for a new run
                # Try a few different restart strategies
                
                # Strategy 1: Perturb the best known point slightly
                # Strategy 2: Random point in a smaller box around the best point
                # We'll use a mix: perturb best point with a moderate step
                perturb_sigma = 1.0
                x_es = np.clip(best_x + perturb_sigma * rng.standard_normal(n), -5, 5)
                fx_es = wrapped_func(x_es)
                sigma = 0.5
                steps_without_improvement = 0
                successes = 0
                steps_in_window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
