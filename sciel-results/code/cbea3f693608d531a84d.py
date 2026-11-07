"""Portfolio optimizer for BBOB f3 (Rastrigin) combining adaptive ES with population-based exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Rastrigin-like functions:
    1. Initial coarse random sampling to identify promising basins.
    2. Adaptive (1+1)-ES with success rule for local refinement.
    3. Population-based exploration with adaptive step sizes and mean-shift.
    4. Time-bounded to stay within wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to find good basins
        n_initial = min(40, max(5, budget // 12))
        for _ in range(n_initial):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)

        if best_x is None:
            x = np.zeros(dim)
            safe_eval(x)
            return best_x, best_f

        # Phase 2: Adaptive (1+1)-ES local refinement from the best point
        if not check_deadline():
            self._run_adaptive_es(func, best_x, best_f, budget - calls_made, rng, check_deadline, safe_eval)

        # Phase 3: Population-based exploration with adaptive step sizes
        if not check_deadline():
            self._run_population_search(func, best_x, best_f, budget - calls_made, rng, check_deadline, safe_eval)

        # Phase 4: Fine-tuning around the best solution found
        if not check_deadline() and best_x is not None:
            self._fine_tune(func, best_x, best_f, min(30, budget - calls_made), rng, check_deadline, safe_eval)

        return best_x, best_f

    def _run_adaptive_es(self, func, x_start, f_start, budget, rng, check_deadline, safe_eval):
        """Adaptive (1+1)-ES with success rule for local refinement."""
        x = x_start.copy()
        f = f_start
        sigma = 1.0
        min_sigma = 1e-5
        max_sigma = 5.0
        successes = 0
        trials = 0

        for _ in range(budget):
            if check_deadline():
                break

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = safe_eval(y)
            trials += 1

            if fy < f:
                x = y
                f = fy
                successes += 1
                sigma *= 1.15
                sigma = min(sigma, max_sigma)
            else:
                sigma *= 0.9
                sigma = max(sigma, min_sigma)

            # Adapt sigma based on success rate over last 15 trials
            if trials % 15 == 0:
                success_rate = successes / 15.0
                if success_rate > 0.3:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.75
                sigma = np.clip(sigma, min_sigma, max_sigma)
                successes = 0

            # If converged, break
            if sigma < min_sigma * 2:
                break

    def _run_population_search(self, func, best_x, best_f, budget, rng, check_deadline, safe_eval):
        """Population-based exploration with adaptive step sizes and mean-shift."""
        pop_size = 8
        # Initialize population around the best point and random points
        population = [best_x.copy()]
        for _ in range(pop_size - 1):
            if rng.random() < 0.5:
                # Perturb current best
                x = best_x + rng.uniform(-2, 2, self.dim)
            else:
                # Random point
                x = rng.uniform(-5, 5, self.dim)
            population.append(np.clip(x, -5, 5))

        # Evaluate initial population
        pop_f = []
        for x in population:
            if check_deadline():
                break
            fx = safe_eval(x)
            pop_f.append(fx)

        # Main loop: evolve the population
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 4.0
        stagnation_counter = 0
        prev_best = best_f

        while not check_deadline():
            # Sort population by fitness
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Check for improvement
            if pop_f[0] < best_f - 1e-10:
                best_f = pop_f[0]
                best_x = population[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnant for too long, reset population
            if stagnation_counter > 20:
                population = [best_x.copy()]
                for _ in range(pop_size - 1):
                    x = best_x + rng.uniform(-1, 1, self.dim)
                    population.append(np.clip(x, -5, 5))
                pop_f = []
                for x in population:
                    if check_deadline():
                        break
                    fx = safe_eval(x)
                    pop_f.append(fx)
                stagnation_counter = 0
                continue

            # Generate offspring
            new_population = []
            new_pop_f = []
            for _ in range(pop_size):
                if check_deadline():
                    break

                # Select parent: weighted by rank
                parent_idx = self._select_parent(pop_f, rng)
                parent = population[parent_idx]

                # Mutate
                child = parent + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, -5, 5)

                fx = safe_eval(child)
                new_population.append(child)
                new_pop_f.append(fx)

            # Combine and select
            combined_pop = population + new_population
            combined_f = pop_f + new_pop_f
            indices = np.argsort(combined_f)
            population = [combined_pop[i] for i in indices[:pop_size]]
            pop_f = [combined_f[i] for i in indices[:pop_size]]

            # Adapt sigma based on success rate
            # Count how many offspring are better than their parent
            success_count = 0
            for i in range(len(new_pop_f)):
                if new_pop_f[i] < pop_f[i]:  # Compare with original parent's fitness
                    success_count += 1

            success_rate = success_count / max(1, len(new_pop_f))
            if success_rate > 0.3:
                sigma *= 1.1
            elif success_rate < 0.1:
                sigma *= 0.85
            sigma = np.clip(sigma, min_sigma, max_sigma)

    def _fine_tune(self, func, best_x, best_f, budget, rng, check_deadline, safe_eval):
        """Fine-tuning around the best solution found."""
        x = best_x.copy()
        f = best_f
        sigma = 0.1
        min_sigma = 1e-6
        max_sigma = 0.5
        successes = 0
        trials = 0

        for _ in range(budget):
            if check_deadline():
                break

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = safe_eval(y)
            trials += 1

            if fy < f:
                x = y
                f = fy
                successes += 1
                sigma *= 1.1
                sigma = min(sigma, max_sigma)
            else:
                sigma *= 0.9
                sigma = max(sigma, min_sigma)

            if trials % 10 == 0:
                success_rate = successes / 10.0
                if success_rate > 0.4:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.8
                sigma = np.clip(sigma, min_sigma, max_sigma)
                successes = 0

            if sigma < min_sigma * 2:
                break

    def _select_parent(self, pop_f, rng):
        """Select a parent using tournament selection."""
        tournament_size = 3
        candidates = rng.choice(len(pop_f), size=tournament_size, replace=False)
        best_candidate = min(candidates, key=lambda i: pop_f[i])
        return best_candidate

# EVOLVE-BLOCK-END
