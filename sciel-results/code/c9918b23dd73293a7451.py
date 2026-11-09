"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Diverse initialization points for Rosenbrock
        init_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
            np.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
            np.array([4.0, 2.0, 0.0, -2.0, -4.0]),
            np.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
            np.array([2.5, 2.5, 2.5, 2.5, 2.5]),
        ]
        # Add random points
        for _ in range(10):
            init_points.append(rng.uniform(-5, 5, n))

        best_x = None
        best_f = np.inf
        calls_used = 0

        # First pass: quick evaluation of all init points
        for init in init_points:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.clip(init, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(n)
            fx = func(x)
            calls_used += 1
            best_f = fx
            best_x = x.copy()

        # Multi-restart local search
        restarts = 0
        max_restarts = 20
        max_no_improve = 30

        while calls_used < budget and restarts < max_restarts:
            if time.time() > deadline:
                break

            # Start from best found so far or a random point
            if restarts == 0:
                x = best_x.copy()
                fx = best_f
            else:
                # Mix: 50% chance from best, 50% from random
                if rng.random() < 0.5:
                    x = best_x + rng.normal(0, 0.5, n)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            sigma = 1.5
            no_improve = 0
            local_calls = 0
            max_local_calls = max(30, (budget - calls_used) // (max_restarts - restarts))

            while calls_used < budget and no_improve < max_no_improve and local_calls < max_local_calls:
                if time.time() > deadline:
                    break

                # Adaptive sigma based on recent success rate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_used += 1
                local_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

            restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
