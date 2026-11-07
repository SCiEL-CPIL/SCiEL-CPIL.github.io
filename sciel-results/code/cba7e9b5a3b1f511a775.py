"""Population-based stochastic evolutionary strategy for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolutionary strategy with adaptive sigma for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(7777)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        def evaluate(x):
            nonlocal calls_used
            x = clip(x)
            f = func(x)
            calls_used += 1
            update_best(x, f)
            return x, f

        # Population-based evolutionary strategy
        # Maintain a population of candidates with Gaussian perturbations
        # and dynamically shrinking sigma to exploit ridge anisotropy

        pop_size = 50
        sigma = 2.0  # Initial step size
        sigma_min = 0.01
        sigma_decay = 0.97  # Multiplicative decay per generation
        elite_count = 10  # Number of elites to keep

        # Initialize population
        population = []
        pop_f = []

        # First few points are diverse
        init_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for p in init_points:
            if not time_ok() or calls_used >= budget - 5:
                break
            x, f = evaluate(p)
            population.append(x)
            pop_f.append(f)

        # Fill rest of population
        while len(population) < pop_size and time_ok() and calls_used < budget - 5:
            if best_x is not None:
                x_try = clip(best_x + rng.normal(0, sigma, dim))
            else:
                x_try = rng.uniform(-5, 5, dim)
            x, f = evaluate(x_try)
            population.append(x)
            pop_f.append(f)

        # Main evolutionary loop
        gen = 0
        while time_ok() and calls_used < budget - 10:
            gen += 1

            # Sort population by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Elites are the first elite_count points
            elites = population[:elite_count]
            elite_f = pop_f[:elite_count]

            # If no improvement in last few generations, shrink sigma more aggressively
            # (implicit through sigma_decay)

            # Generate offspring
            new_population = []
            new_pop_f = []

            # Keep elites
            for i in range(min(elite_count, len(population))):
                new_population.append(population[i])
                new_pop_f.append(pop_f[i])

            # Generate new candidates
            while len(new_population) < pop_size and time_ok() and calls_used < budget - 5:
                # Pick a parent (biased toward elites)
                if rng.random() < 0.7 and len(elites) > 0:
                    parent = elites[rng.integers(len(elites))]
                else:
                    parent = population[rng.integers(len(population))]

                # Gaussian perturbation
                x_try = clip(parent + rng.normal(0, sigma, dim))
                x, f = evaluate(x_try)
                new_population.append(x)
                new_pop_f.append(f)

            # Update population
            population = new_population
            pop_f = new_pop_f

            # Adapt sigma
            # If best improved recently, keep sigma; otherwise shrink
            sigma = max(sigma * sigma_decay, sigma_min)

            # Occasionally do a local refinement around the best point
            if gen % 5 == 0 and best_x is not None and time_ok() and calls_used < budget - 10:
                # Local search around best with smaller steps
                for _ in range(5):
                    if not time_ok() or calls_used >= budget - 5:
                        break
                    x_try = clip(best_x + rng.normal(0, max(sigma * 0.3, 0.05), dim))
                    x, f = evaluate(x_try)

        # Final polish: coordinate descent on best point
        if best_x is not None and time_ok() and calls_used < budget - 5:
            x = best_x.copy()
            f_curr = best_f
            for _ in range(10):
                if not time_ok() or calls_used >= budget - 3:
                    break
                improved = False
                for i in range(dim):
                    if not time_ok() or calls_used >= budget - 3:
                        break
                    for step in [0.1, 0.05, 0.2, 0.01]:
                        for sign in [1.0, -1.0]:
                            if not time_ok() or calls_used >= budget - 3:
                                break
                            x_try = clip(x + sign * step * np.eye(dim)[:, i])
                            f_try = func(x_try)
                            calls_used += 1
                            if f_try < f_curr - 1e-12:
                                x = x_try
                                f_curr = f_try
                                improved = True
                                update_best(x, f_curr)
                                break
                if not improved:
                    break

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
