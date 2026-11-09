import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget or time.time() >= deadline:
                return
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Exploration
        # Use a mix of Halton-like and random points to cover the space
        global_budget = int(budget * 0.3)

        # Start with center
        evaluate(np.zeros(dim))

        # Simple Halton sequence for low-discrepancy sampling
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        idx = 1
        while calls < global_budget and time.time() < deadline - 10:
            point = np.array([halton(idx + i, primes[i]) for i in range(dim)])
            point = point * 10.0 - 5.0
            evaluate(point)
            idx += 1

        if calls >= budget or best_x is None:
            return best_x, best_f

        # Phase 2: Local Refinement using (1+1)-ES with adaptive sigma
        remaining_budget = budget - calls

        # Run (1+1)-ES from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size

        # 1/5th success rule for sigma adaptation
        successes = 0
        trials = 0
        target_success_rate = 1.0 / 5.0
        sigma_min = 1e-8
        sigma_max = 5.0

        # Also implement a restart mechanism if stagnation occurs
        stagnation_limit = 50
        no_improve = 0

        while calls < budget and time.time() < deadline - 2:
            if no_improve > stagnation_limit:
                # Restart with a larger sigma or new random point
                # If we've been stuck, try a random jump
                if fx < best_f - 1e-10:
                    # We are better than global best, maybe just increase sigma
                    sigma *= 2.0
                    sigma = min(sigma, sigma_max)
                    no_improve = 0
                else:
                    # Try a random restart from a new location
                    x = rng.uniform(-5, 5, size=dim)
                    fx = evaluate(x)
                    if fx is None:
                        break
                    sigma = 1.0
                    no_improve = 0
                    successes = 0
                    trials = 0
                    continue

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            trials += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                no_improve = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                no_improve += 1

            # Adapt sigma every 10 trials using 1/5th success rule
            if trials >= 10:
                success_rate = successes / trials
                if success_rate > target_success_rate:
                    sigma *= 1.5
                elif success_rate < target_success_rate:
                    sigma /= 1.5

                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = 0
                trials = 0

        return best_x, best_f
