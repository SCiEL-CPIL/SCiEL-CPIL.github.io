"""Broad exploration with multi-start (1+1)-ES for Rastrigin f15."""
import numpy as np
import time
import math


class Optimizer:
    """Broad exploration followed by adaptive (1+1)-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Broad exploration with random starts
        n_random = 15
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Multi-start (1+1)-ES with step-size-based restarts
        # For each random start (including the best found), run a (1+1)-ES
        # that adapts the step size based on success rate.
        
        def run_1plus1_es(x0, sigma0, max_evals):
            """Run (1+1)-ES from x0 with initial sigma0."""
            nonlocal calls_used, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            f_x = best_f  # We already know f at best_x from exploration
            if not np.allclose(x, best_x):
                f_x = evaluate(x)
            
            no_improve = 0
            evals_this_run = 0
            
            while evals_this_run < max_evals and time_ok():
                # Generate trial
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = evaluate(x_trial)
                evals_this_run += 1
                
                if f_trial < f_x:
                    # Success: move to trial, increase sigma
                    x = x_trial.copy()
                    f_x = f_trial
                    no_improve = 0
                    sigma *= 1.1  # Increase step size on success
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9
                    no_improve += 1
                
                # Restart if no improvement for too long or sigma too small
                if no_improve > 20 or sigma < 0.01:
                    break
                
                # Keep global best updated
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
            
            return best_f

        # Try (1+1)-ES from the current best with different initial sigmas
        sigmas_to_try = [1.0, 0.5, 0.2, 0.1]
        for sigma0 in sigmas_to_try:
            if not time_ok():
                break
            remaining = budget - calls_used
            if remaining < 20:
                break
            max_evals = min(remaining, 100)
            run_1plus1_es(best_x.copy(), sigma0, max_evals)

        # Phase 3: If budget remains, do more random restarts with (1+1)-ES
        while time_ok() and calls_used < budget - 10:
            x0 = rng.uniform(-5, 5, dim)
            sigma0 = rng.uniform(0.3, 1.5)
            remaining = budget - calls_used
            max_evals = min(remaining, 50)
            if max_evals < 5:
                break
            run_1plus1_es(x0, sigma0, max_evals)

        return (best_x, best_f)
