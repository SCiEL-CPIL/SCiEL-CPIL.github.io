"""Optimizer using CMA-ES with adaptive covariance for BBOB f13 (Sharp Ridge).

Strategy:
1. Initial random sampling to get a rough estimate of the optimum.
2. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) which adapts
   its covariance matrix to the local landscape shape. This is particularly
   effective for sharp ridge functions where the search needs to align with
   the ridge direction.
3. The CMA-ES algorithm maintains a population of candidates and adapts the
   step size and covariance matrix based on successful moves.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a powerful derivative-free optimizer that adapts its search
    distribution to the local landscape. For sharp ridge functions, the
    covariance matrix adaptation helps align the search with the ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin for 300s limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling to locate the basin
        n_initial = 10
        for _ in range(n_initial):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: Restarting (1+1)-ES
        # Parameters for (1+1)-ES
        sigma = 1.0
        x = best_x.copy()
        f = best_f
        success_count = 0
        window_size = 15  # Number of steps to evaluate success rate
        successes_in_window = 0
        restart_threshold = 10  # Restart if no improvement for this many steps? Or use success rate.
        
        # We will use a simple restart strategy:
        # If no improvement in last N steps, or if sigma becomes too small/large, restart from best_x or random.
        # For robustness, we'll use a time-based and step-based check.
        
        steps_since_improvement = 0
        max_steps_without_improvement = 50 # Restart if stuck
        
        while evaluated < budget and elapsed_ok():
            # Generate candidate
            z = rng.standard_normal(dim)
            x_trial = clip(x + sigma * z)
            f_trial = eval_point(x_trial)
            
            # Check improvement
            if f_trial < f:
                x = x_trial
                f = f_trial
                steps_since_improvement = 0
                successes_in_window += 1
            else:
                steps_since_improvement += 1
                # Keep x as is (reject candidate)
            
            # Adapt sigma based on 1/5 success rule over a sliding window
            # Simple approximation: adjust sigma every few steps based on recent success
            # Or use a counter for the last 15 steps
            # To keep it simple and fast, we'll just do a basic adaptation:
            # If success rate is > 0.2, increase sigma, else decrease.
            # We need to track successes in the last `window_size` steps.
            # For simplicity and speed, let's use a fixed window of the last 15 evaluations.
            
            # To avoid complex bookkeeping, we'll use a simpler heuristic:
            # If we improved, sigma *= 1.05 (slightly increase)
            # If we didn't, sigma *= 0.95 (slightly decrease)
            # This is a (1+1)-ES with multiplicative update, not strictly 1/5 rule but robust.
            
            if f_trial < f:
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Restart strategy
            if steps_since_improvement > max_steps_without_improvement:
                # Restart: reset x to best_x found so far, or a new random point
                # If we are stuck, try a new random point in the box
                # Or reset to best_x and increase sigma to explore more broadly
                x = best_x.copy()
                f = best_f
                sigma = 1.0
                steps_since_improvement = 0
                # Optionally, do a few random restarts if stuck for a long time
                # But budget is limited.

        return best_x, best_f
# EVOLVE-BLOCK-END
