import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = float('inf')
        calls = 0

        def check_deadline():
            return time.time() > deadline

        def record_best(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        def run_population_es(x_start, fx_start, max_calls, pop_size=4, sigma_init=1.0):
            """Population-based local search with batched offspring."""
            nonlocal calls, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-6
            max_sigma = 5.0
            no_improve = 0
            max_no_improve = 20

            for _ in range(max_calls):
                if calls >= budget:
                    break
                if check_deadline():
                    break

                # Check if we have enough budget for the batch
                if calls + pop_size > budget:
                    # Do a single step if we can't fit the batch
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    record_best(y, fy)
                    if fy < fx:
                        x = y
                        fx = fy
                        no_improve = 0
                    else:
                        no_improve += 1
                    break

                # Generate batch of offspring
                offspring = x + sigma * rng.standard_normal((pop_size, dim))
                offspring = np.clip(offspring, -5, 5)

                best_offspring = None
                best_offspring_f = float('inf')

                for y in offspring:
                    fy = func(y)
                    calls += 1
                    record_best(y, fy)
                    if fy < best_offspring_f:
                        best_offspring_f = fy
                        best_offspring = y

                if best_offspring_f < fx:
                    x = best_offspring
                    fx = best_offspring_f
                    sigma = min(sigma * 1.2, max_sigma)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.8, min_sigma)
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break

            return x, fx

        def run_fine_es(x_start, fx_start, max_calls, sigma_init=0.5, window=10):
            """(1+1)-ES with windowed 1/5 success rule for fine-tuning."""
            nonlocal calls, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-8
            max_sigma = 2.0
            successes = 0

            for _ in range(max_calls):
                if calls >= budget:
                    break
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                record_best(y, fy)

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1

                if calls % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

            return x, fx

        # Phase 1: Grid-based coarse search
        # Use a 3^5 grid over [-5, 5]^5 (243 points)
        # This is cheap and systematic
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([-5.0, 0.0, 5.0][i], dtype=float)
                            x[1] = np.array([-5.0, 0.0, 5.0][j])
                            x[2] = np.array([-5.0, 0.0, 5.0][k])
                            x[3] = np.array([-5.0, 0.0, 5.0][l])
                            x[4] = np.array([-5.0, 0.0, 5.0][m])
                            grid_points.append(x)

        # Only evaluate if we have enough budget
        grid_budget = min(len(grid_points), budget // 2)
        for x in grid_points[:grid_budget]:
            if calls >= budget:
                break
            if check_deadline():
                break
            fx = func(x)
            calls += 1
            record_best(x, fx)

        # Phase 2: Population-based local search from best grid point
        if global_best_x is not None and calls < budget:
            remaining = budget - calls
            # Use 60% of remaining budget for population ES
            pop_budget = int(remaining * 0.6)
            if pop_budget > 0:
                run_population_es(global_best_x, global_best_f, pop_budget, pop_size=4, sigma_init=1.0)

        # Phase 3: Fine-tuning with (1+1)-ES
        if global_best_x is not None and calls < budget and not check_deadline():
            remaining = budget - calls
            run_fine_es(global_best_x, global_best_f, remaining, sigma_init=0.5, window=10)

        return global_best_x, global_best_f
