import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, keep 20s buffer
        deadline = start_time + 280
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline
        
        # State
        best_x = np.random.uniform(-5, 5, d)
        best_f = func(best_x)
        calls = 1
        
        # Parameters for the random walk
        # Start with a large step to explore the [-5, 5]^5 box quickly
        # The box side is 10, so a step of 2.0 covers a significant portion
        current_sigma = 2.0 
        min_sigma = 1e-6
        decay_factor = 0.995
        
        # Track progress to adjust sigma
        # We want to make progress early. 
        # If we find a better point, we can afford to shrink sigma slightly or keep it.
        # If we don't, we might need to jump more.
        
        # A simple heuristic:
        # 1. Generate a candidate around the current best.
        # 2. If it improves, accept and shrink sigma.
        # 3. If it doesn't, reject and keep sigma (or grow it slightly if stuck).
        
        # However, for AOCC, early progress is king.
        # A "Lévy Flight" or heavy-tailed distribution might be better for exploration.
        # But let's stick to a Gaussian with aggressive initial size and decay.
        
        # Alternative: Multi-start with random restarts if stuck.
        # Let's implement a hybrid: 
        # - Start with a random walk from a random point.
        # - If no improvement for N steps, restart from a new random point in the box.
        # - This ensures we don't get stuck in a local minimum (though f11 is unimodal, 
        #   the rotation/shift makes it look different, and "Discus" has a specific shape).
        
        # Actually, f11 (Discus) is unimodal. The main risk is the scale.
        # The minimum is in [-4, 4]^5.
        # A random walk with sigma=2.0 is good.
        
        # Let's refine the strategy:
        # 1. Initial point: Uniform in [-5, 5]^d.
        # 2. Loop:
        #    - Generate x_new = best_x + sigma * N(0, I)
        #    - Clip to [-5, 5]
        #    - Evaluate
        #    - If f(x_new) < best_f:
        #        - best_x, best_f = x_new, f(x_new)
        #        - sigma *= decay_factor (converge)
        #    - Else:
        #        - # No improvement. 
        #        - # To avoid stagnation, if we have many consecutive failures, 
        #        - # we might want to increase sigma or restart.
        #        - # But for unimodal functions, shrinking sigma is usually safe if we are close.
        #        - # If we are far, we need to move more.
        #        - # Let's use a fixed number of failures to trigger a "jump" or restart.
        
        # Given the budget is only 500, and d=5.
        # 500 calls is not a lot.
        # CMA-ES is good but might be overkill or slow to start.
        # A simple Nelder-Mead might work but is fragile.
        # Random walk with adaptive sigma is robust.
        
        # Let's try a "Simulated Annealing" like approach or just a simple hill climbing.
        # Hill climbing:
        # sigma starts at 2.0.
        # If improvement: sigma *= 0.95 (shrink)
        # If no improvement: sigma *= 1.05 (expand slightly, up to max)
        # Max sigma: 5.0 (covers the box)
        
        max_sigma = 5.0
        
        while calls < budget and time_ok():
            # Generate candidate
            z = np.random.randn(d)
            x_candidate = best_x + current_sigma * z
            x_candidate = np.clip(x_candidate, -5, 5)
            
            f_candidate = func(x_candidate)
            calls += 1
            
            if f_candidate < best_f:
                best_f = f_candidate
                best_x = x_candidate.copy()
                # Shrink step size as we get closer to the minimum
                current_sigma *= 0.95
            else:
                # No improvement
                # Expand step size to escape potential local minima or if we are far
                # But be careful not to oscillate.
                current_sigma *= 1.02
                if current_sigma > max_sigma:
                    current_sigma = max_sigma
            
            # Optional: Restart logic
            # If we haven't improved in, say, 50 steps, and sigma is small, 
            # we might be stuck. Restart from a random point?
            # For unimodal f11, this is unlikely to be necessary if sigma adapts well.
            # But if the initial point was bad and we converged to a plateau?
            # f11 is smooth.
            
            # Let's add a restart counter
            # If no improvement for 30 steps, restart with a new random center?
            # Actually, let's just keep it simple first.
            
        return best_x, best_f
