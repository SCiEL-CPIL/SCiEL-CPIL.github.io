"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # (1+1)-ES with 1/5 success rule
        # Initial step size: cover the search space
        sigma = 3.0
        
        # Start from origin (Schwefel-like functions often have global min near origin after transformations)
        x = np.zeros(dim)
        try:
            f = func(x)
        except Exception:
            f = np.inf
        
        best_x = x.copy()
        best_f = f
        
        evals = 1
        successes = 0
        total_attempts = 0
        
        def time_exceeded():
            return time.time() > deadline
        
        while evals < budget and not time_exceeded():
            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            np.clip(candidate, -5, 5, out=candidate)
            
            try:
                f_candidate = func(candidate)
            except Exception:
                f_candidate = np.inf
            
            evals += 1
            total_attempts += 1
            
            if f_candidate < f:
                x = candidate
                f = f_candidate
                successes += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # Candidate rejected, x stays the same
                pass
            
            # Update sigma using 1/5 success rule
            # Target success rate is 1/5 = 0.2
            # If success rate > 0.2, increase sigma; if < 0.2, decrease sigma
            # Use a simple multiplicative update
            success_rate = successes / max(1, total_attempts)
            if success_rate > 0.2:
                sigma *= 1.05
            elif success_rate < 0.2:
                sigma *= 0.95
            
            # Reset success counter every 10 attempts for responsiveness
            if total_attempts % 10 == 0:
                successes = 0
                total_attempts = 0
            
            # If sigma becomes too small, we've likely converged
            if sigma < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
