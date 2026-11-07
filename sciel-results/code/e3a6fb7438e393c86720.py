"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES with adaptive step size for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-Start Exploration
        # Evaluate a small number of random points to find a good starting region.
        # This captures the "population diversity" aspect of DE but with lower overhead.
        n_starts = 10
        if budget < n_starts + 10:
            n_starts = max(1, budget // 10)
        
        candidates = rng.uniform(-5, 5, (n_starts, dim))
        candidate_f = np.array([func(c) for c in candidates])
        budget_used = n_starts

        best_idx = np.argmin(candidate_f)
        x = candidates[best_idx].copy()
        fx = candidate_f[best_idx]

        # Phase 2: Local Refinement with (1+1)-ES
        # Use the remaining budget for adaptive local search.
        # Inspired by the high-scoring (1+1)-ES implementations.
        sigma = 1.5
        success_window = []
        window_size = 10
        min_sigma = 0.01
        max_sigma = 10.0

        while budget_used < budget:
            if time.time() > deadline:
                break
            
            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            # Update best and track success
            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
            else:
                success_window.append(0)
            
            # Adapt sigma based on 1/5th success rule
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.2:
                    sigma *= 1.05  # Increase step size if too many successes
                elif success_rate < 0.1:
                    sigma *= 0.95  # Decrease step size if too few successes
                # Keep window manageable
                if len(success_window) > window_size:
                    success_window.pop(0)
                
                sigma = max(min_sigma, min(sigma, max_sigma))

        return x, fx
# EVOLVE-BLOCK-END
