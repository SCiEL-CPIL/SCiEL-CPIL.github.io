"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Two-phase (1+1)-ES.
    Phase 1: Small random search (10-20 evals) to identify a promising basin.
    Phase 2: Sustained (1+1)-ES with adaptive step size to converge deeply 
             within that basin. Rastrigin's local minima are unimodal, so 
             a single deep lineage outperforms fragmented multi-starts 
             when the basin is correctly identified.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Basin Identification ---
        # Sample a small number of points to find a local minimum.
        # Since the global minimum is in [-4, 4]^5 and the box is [-5, 5]^5,
        # uniform sampling has a reasonable chance of hitting a basin.
        # We use 15 points to keep overhead low but coverage decent.
        n_init = 15
        for _ in range(n_init):
            if calls_made >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)

        # --- Phase 2: Deep Convergence via (1+1)-ES ---
        # Start from the best point found.
        x = best_x.copy()
        fx = best_f
        
        # Adaptive step size
        # Start with a smallish step to stay within the basin, but large enough to explore.
        # Rastrigin has local minima width ~ 2*pi/sqrt(10) approx? 
        # Actually, the local minima of Rastrigin are at integers. Distance between is 2.
        # A step size of 0.5 to 1.0 is reasonable for local search.
        # Let's start at 0.5 and adapt.
        sigma = 0.5
        
        # (1+1)-ES Parameters
        # Success rate control: if success rate > 0.2, increase sigma; else decrease.
        # Standard 1/5th success rule or similar.
        # Let's use a simple adaptive rule:
        # - Keep a window of last 10 trials.
        # - If > 2 successes (improvements), sigma *= 1.2
        # - If < 1 success, sigma /= 1.2
        
        trial_history = [] # 1 for success, 0 for fail
        window_size = 10
        
        # Loop until budget or deadline
        while calls_made < budget and time.time() < deadline:
            # Generate offspring
            z = rng.standard_normal(dim)
            x_child = x + sigma * z
            x_child = np.clip(x_child, -5.0, 5.0)
            
            f_child = func(x_child)
            calls_made += 1
            
            # Update global best
            update_best(x_child, f_child)
            
            # Update local state for (1+1)-ES
            success = 0
            if f_child < fx:
                success = 1
                x = x_child
                fx = f_child
            
            # Adaptive sigma
            trial_history.append(success)
            if len(trial_history) > window_size:
                trial_history.pop(0)
            
            # Adjust sigma based on success rate in window
            if len(trial_history) == window_size:
                success_rate = np.mean(trial_history)
                if success_rate > 0.2:
                    sigma *= 1.15
                elif success_rate < 0.1:
                    sigma /= 1.15
                
                # Keep sigma within reasonable bounds
                sigma = np.clip(sigma, 1e-4, 5.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
