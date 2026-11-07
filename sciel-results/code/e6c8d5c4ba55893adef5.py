"""Multi-start (1+1)-ES with momentum for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_f = np.inf
        evals_used = 0
        
        def time_left():
            return timeout - (time.time() - start_time)

        def evaluate(x):
            nonlocal evals_used, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start (1+1)-ES with momentum
        # Strategy: Use diverse starting points, each with its own (1+1)-ES
        # that has momentum to escape local minima.
        
        # Number of restarts: we'll use a fixed number based on budget
        # For budget=500, d=5, we can afford ~50-100 evaluations per restart
        n_restarts = 8
        evals_per_restart = budget // n_restarts
        
        # Starting points: mix of random and structured
        rng = np.random.RandomState(42)
        
        # Generate diverse starting points
        start_points = []
        # Include origin and some structured points
        start_points.append(np.zeros(d))
        # Add points near corners of the box
        for _ in range(min(3, n_restarts - 1)):
            pt = rng.uniform(-4, 4, size=d)
            start_points.append(pt)
        # Fill remaining with random points
        while len(start_points) < n_restarts:
            start_points.append(rng.uniform(-4.5, 4.5, size=d))
        
        for start_idx, start_pt in enumerate(start_points):
            if evals_used >= budget or time_left() < 5.0:
                break
            
            # Evaluate starting point
            f_start = evaluate(start_pt)
            
            # (1+1)-ES parameters for this restart
            x = start_pt.copy()
            f_x = f_start
            sigma = 1.5  # Initial step size
            momentum = np.zeros(d)
            momentum_coeff = 0.7  # Momentum coefficient
            
            # Track best for this restart
            local_best_f = f_x
            local_best_x = x.copy()
            
            # Maximum evaluations for this restart
            max_evals_this_restart = min(evals_per_restart, budget - evals_used)
            evals_this_restart = 1  # Already evaluated start point
            
            while evals_this_restart < max_evals_this_restart and evals_used < budget and time_left() > 2.0:
                # Generate candidate step with momentum
                step = np.random.randn(d)
                # Add momentum from previous successful direction
                if evals_this_restart > 1:
                    step = momentum_coeff * momentum + np.sqrt(1 - momentum_coeff**2) * step
                
                y = x + sigma * step
                y = np.clip(y, -5, 5)
                f_y = evaluate(y)
                evals_this_restart += 1
                
                if f_y < f_x:
                    # Success: update position and momentum
                    momentum = step / (np.linalg.norm(step) + 1e-10)
                    x = y
                    f_x = f_y
                    # Slightly increase sigma on success
                    sigma *= 1.03
                    if x < local_best_x and f_y < local_best_f:
                        local_best_f = f_y
                        local_best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.95
                    # Decay momentum
                    momentum *= 0.5
                
                # If sigma becomes too small, break
                if sigma < 1e-4:
                    break
                
                # If we've made good progress, we can continue
                # If stuck for many iterations, break early to save budget
                if evals_this_restart > 10 and f_x - local_best_f < 1e-6 and sigma < 0.1:
                    break
            
            # After each restart, if we found a good point, use it as reference
            # for the next restart's starting point neighborhood
            if start_idx < n_restarts - 1 and best_f < np.inf:
                # Perturb the best found so far for next start
                next_start = best_x + np.random.randn(d) * 0.5
                next_start = np.clip(next_start, -4.5, 4.5)
                # Replace the next start point with this perturbed best
                # This creates a chain of exploration from good regions
                if start_idx + 1 < len(start_points):
                    start_points[start_idx + 1] = next_start
        
        return best_x, best_f
# EVOLVE-BLOCK-END
