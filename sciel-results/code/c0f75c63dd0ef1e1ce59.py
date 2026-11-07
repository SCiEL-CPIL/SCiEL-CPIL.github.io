"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive random walk with 10 parallel lines."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initialize 10 independent search lines
        n_lines = 10
        # Spread initial points across the box to maximize coverage
        xs = np.empty((n_lines, dim))
        for i in range(n_lines):
            xs[i] = rng.uniform(-5, 5, dim)
        
        fxs = np.empty(n_lines)
        for i in range(n_lines):
            fxs[i] = func(xs[i])
        
        calls = n_lines
        sigmas = np.ones(n_lines) * 2.0

        # Track the global best
        best_idx = np.argmin(fxs)
        global_best_x = xs[best_idx].copy()
        global_best_f = fxs[best_idx]

        # Main loop: distribute remaining budget among lines
        # We will do rounds of n_lines evaluations until budget is exhausted
        # This ensures balanced exploration across all lines
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Select a line to update. 
            # Strategy: Round-robin to ensure all lines get equal chance,
            # but we can bias towards better lines if we want. 
            # For simplicity and robustness, let's do round-robin.
            # Actually, a simple sequential update of all lines in a loop is fine.
            
            for i in range(n_lines):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate candidate
                # Adaptive sigma: expand if improvement, shrink if not
                # Add a small random jitter to sigma to avoid getting stuck in local minima
                sigma = sigmas[i]
                y = xs[i] + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < fxs[i]:
                    # Improvement on this line
                    xs[i] = y
                    fxs[i] = fy
                    sigmas[i] *= 1.5
                else:
                    # No improvement
                    sigmas[i] *= 0.85
                    # Occasional large kick to escape local minima
                    if rng.random() < 0.01:
                        sigmas[i] = 2.0
                        # Also restart position slightly? No, just step size.
                        # Or maybe jump to a random point near current best?
                        # Let's stick to step size adjustment for now.

                # Update global best
                if fxs[i] < global_best_f:
                    global_best_f = fxs[i]
                    global_best_x = xs[i].copy()
                    best_idx = i

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
