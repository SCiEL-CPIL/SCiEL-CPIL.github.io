import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(12345)

        x = np.zeros(dim)
        fx = func(x)

        best_x = x.copy()
        best_f = fx

        sigma = 1.5

        c_inc = 1.08
        c_dec = 1.0 / 1.08

        calls_made = 1

        window = 10
        successes = 0
        step_count = 0

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                step_count += 1

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                step_count += 1

            # Adapt sigma every 'window' steps
            if step_count % window == 0:
                if successes / window > 1.0 / (window + 1):
                    sigma *= c_inc
                else:
                    sigma *= c_dec
                successes = 0

            # If sigma becomes too small, we might be stuck; add a small restart chance
            if sigma < 1e-6:
                if rng.random() < 0.01:
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 1.0
                else:
                    sigma = 1e-6

        return best_x, best_f
