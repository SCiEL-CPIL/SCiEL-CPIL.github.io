"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s wall clock limit. Keep a safety margin of 5s.
        deadline = start_time + 325.0
        
        best_x = None
        best_f = np.inf
        call_count = 0
        n = self.dim
        
        def make_call(x):
            nonlocal call_count
            if call_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            call_count += 1
            return fx

        # 1. Initial Probe
        # Start at origin. It's a safe, central point.
        x0 = np.zeros(n)
        f0 = make_call(x0)
        if f0 is not None and np.isfinite(f0):
            best_f = f0
            best_x = x0.copy()
        else:
            # Fallback if origin fails (unlikely but safe)
            return None, np.inf

        # 2. Strategy: (1+1)-ES with 1/5th rule step-size adaptation
        # This is a fundamentally different approach from Simulated Annealing.
        # It adapts the step size based on success rate, which is more robust
        # for ill-conditioned problems like Schaffers F7.
        
        # Parameters for (1+1)-ES
        sigma = 1.0  # Initial step size
        c_sigma = 1.0 / (n + 4)  # Learning rate for sigma
        # 1/5th rule: if success rate > 1/5, increase sigma; else decrease
        # We'll use a simple counter-based approach or the standard update
        
        # To handle the "rotation" and "shift", we rely on the isotropic 
        # nature of the Gaussian perturbation. The 1/5 rule will adapt the 
        # scale.
        
        # Multi-start: We will run several short (1+1)-ES runs from different 
        # starting points, or restart if stagnation is detected.
        # Given the budget of 500, we can afford a few restarts.
        
        # Let's implement a single (1+1)-ES with adaptive restarts.
        # If no improvement for K steps, restart from best_x with a smaller sigma?
        # Or restart from a random point?
        # Since we have a global best, restarting from best_x with a fresh sigma 
        # might be good. But if we are stuck in a local minimum, we should explore.
        
        # Let's try: 
        # - Run (1+1)-ES.
        # - If stagnation (no best update) for 50 steps, pick a new random start 
        #   point (but keep global best) and reset sigma to 1.0.
        
        x = best_x.copy()
        f = best_f
        
        stagnation_limit = 30
        stagnation_count = 0
        max_restarts = 5
        restarts_used = 0
        
        while call_count < self.budget and time.time() < deadline:
            # Generate neighbor
            delta = np.random.normal(0, sigma, n)
            x_new = x + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = make_call(x_new)
            if f_new is None:
                break
                
            if not np.isfinite(f_new):
                f_new = f + 1000.0 
                
            # Acceptance and step-size update
            if f_new < f:
                # Success
                x = x_new
                f = f_new
                # Update sigma: increase if success rate is high
                # Simple 1/5 rule update:
                # We need a window. Let's use a simple exponential moving average of success.
                # Or just: if success, sigma *= 1.01; if fail, sigma /= 1.01?
                # Standard (1+1)-ES with 1/5 rule uses a larger window.
                # Let's use a simpler adaptation:
                # sigma = sigma * (1 + c_sigma * (success - 1/5))
                # But we need a success indicator.
                
                # Let's stick to a robust heuristic:
                # If success, slightly increase sigma.
                sigma *= 1.05
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    stagnation_count = 0
            else:
                # Failure
                # Slightly decrease sigma
                sigma /= 1.05
                stagnation_count += 1
                
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Restart logic
            if stagnation_count >= stagnation_limit:
                if restarts_used < max_restarts:
                    # Restart from a random point in the box
                    # This helps escape local minima.
                    x = np.random.uniform(-5, 5, n)
                    f = make_call(x)
                    if f is None:
                        break
                    if not np.isfinite(f):
                        f = best_f
                        x = best_x.copy()
                    else:
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    
                    sigma = 1.0
                    stagnation_count = 0
                    restarts_used += 1
                else:
                    # No more restarts, break if sigma is very small
                    if sigma < 1e-5:
                        break

        # 3. Final Polish
        # If we have budget left, do a few small steps from the best point.
        # This is essentially what the low-sigma (1+1)-ES does, but let's ensure 
        # we don't waste calls if we're already very precise.
        
        if best_x is None:
            # Fallback
            best_x = x0
            best_f = f0

        return best_x, best_f
# EVOLVE-BLOCK-END
