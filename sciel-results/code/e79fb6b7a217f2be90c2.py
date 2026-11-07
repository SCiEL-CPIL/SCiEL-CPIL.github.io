"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size, multi-start, periodic restarts, and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        d = self.dim

        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = self.budget

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find a good initial basin
        # Use diverse starting points: origin, corners, random
        starts = [np.zeros(d)]
        # Add corners and edge midpoints for better coverage
        for i in range(d):
            for val in [-5, 5]:
                pt = np.zeros(d)
                pt[i] = val
                starts.append(pt)
        # Add random points
        for _ in range(6):
            starts.append(rng.uniform(-5, 5, d))

        # Evaluate all starts
        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = s_clipped.copy()

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: (1+1)-ES with adaptive sigma and periodic restarts
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # start with moderate step size
        no_improve_count = 0
        max_no_improve = 30  # restart if stuck for this many iterations

        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = try_eval(y)
            if fy is None:
                break

            if fy < fx:
                x, fx = y, fy
                no_improve_count = 0
                # Increase sigma on success (but cap it)
                sigma = min(sigma * 1.1, 5.0)
            else:
                no_improve_count += 1
                # Decrease sigma on failure
                sigma *= 0.9

            # Periodic restart if stuck
            if no_improve_count >= max_no_improve:
                # Try a new random start
                new_x = rng.uniform(-5, 5, d)
                new_fy = try_eval(new_x)
                if new_fy is None:
                    break
                if new_fy < fx:
                    x, fx = new_x, new_fy
                else:
                    # Keep current best, but reset sigma and counter
                    x = best_x.copy()
                    fx = best_f
                sigma = 1.0
                no_improve_count = 0

            # If sigma is very small, try a larger perturbation
            if sigma < 0.05:
                x = np.clip(x + 0.3 * rng.standard_normal(d), -5, 5)
                fx = try_eval(x)
                if fx is None:
                    break
                sigma = 0.3

        # Phase 3: Polish phase - use very small sigma for fine-tuning
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            polish_calls = min(20, max_calls - calls_used)
            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.8  # shrink further on success
                else:
                    sigma *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
