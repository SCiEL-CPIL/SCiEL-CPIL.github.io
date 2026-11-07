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

        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Population-based Exploration
        pop_size = 12
        sigma = 1.5

        # Initialize population with diverse points including center
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))

        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Evolve population for a few generations
        max_gens = 8
        for gen in range(max_gens):
            if calls >= budget or time.time() > deadline:
                break

            # Sort population
            indices = np.argsort(pop_f)
            population = [population[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Keep top half, generate new half
            half = max(1, pop_size // 2)
            new_population = [population[i].copy() for i in range(half)]
            new_f = pop_f[:half].copy()

            for _ in range(pop_size - half):
                if calls >= budget or time.time() > deadline:
                    break
                # Tournament selection from top half
                parent_idx = rng.integers(0, half)
                parent = population[parent_idx]
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                new_population.append(child)

            # Evaluate new individuals
            for x in new_population[half:]:
                if calls >= budget or time.time() > deadline:
                    break
                new_f.append(eval_func(x))

            population = new_population
            pop_f = new_f

            # Adapt sigma
            sigma *= 0.97
            sigma = np.clip(sigma, 0.1, 4.0)

        # Phase 2: (1+1)-ES Exploitation
        best_idx = np.argmin(pop_f)
        x = population[best_idx].copy()
        f = pop_f[best_idx]

        sigma_es = 0.8
        max_es_calls = min(150, budget - calls - 80)

        for _ in range(max_es_calls):
            if calls >= budget or time.time() > deadline:
                break

            y = x + sigma_es * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = eval_func(y)

            if fy < f:
                x = y
                f = fy
                sigma_es *= 1.05
            else:
                sigma_es *= 0.92

            sigma_es = np.clip(sigma_es, 0.01, 5.0)

            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 3: Rastrigin-aware Lattice Search
        # The function f15 is Rastrigin rotated. Rastrigin has minima at integer points
        # in the rotated coordinate frame. We can generate candidate points near the
        # current best by rounding to nearby integers and evaluating them.
        if calls < budget and time.time() < deadline and best_x is not None:
            x_local = best_x.copy()
            f_local = best_f

            # Strategy A: Coordinate-wise refinement with decreasing step
            step = 0.3
            for _ in range(30):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

            # Strategy B: Snapping to nearby integer-like points
            # Generate candidates by rounding each coordinate to nearest integer ± offset
            if calls < budget and time.time() < deadline:
                candidates = []
                # All combinations of rounding each coordinate to nearest integer with offsets -1, 0, 1
                # This is 3^5 = 243 candidates, which is too many. Let's be smarter.
                # Try rounding the whole vector first, then perturb individual coordinates
                base_rounded = np.clip(np.round(x_local), -5, 5)
                candidates.append(base_rounded)

                # Perturb each coordinate by -1, 0, 1
                for d in range(dim):
                    for offset in [-1, 1]:
                        y = base_rounded.copy()
                        y[d] += offset
                        y = np.clip(y, -5, 5)
                        candidates.append(y)

                # Also try points where only some coordinates are rounded
                for d in range(dim):
                    for e in range(d + 1, dim):
                        y = x_local.copy()
                        y[d] = round(y[d])
                        y[e] = round(y[e])
                        y = np.clip(y, -5, 5)
                        candidates.append(y)

                # Deduplicate and evaluate
                seen = set()
                for y in candidates:
                    if calls >= budget or time.time() > deadline:
                        break
                    key = tuple(np.round(y, 6))
                    if key not in seen:
                        seen.add(key)
                        fy = eval_func(y)
                        if fy < f_local:
                            f_local = fy
                            x_local = y.copy()

            # Strategy C: Final fine-tuning with very small steps
            if calls < budget and time.time() < deadline:
                if f_local < best_f:
                    best_x = x_local.copy()
                    best_f = f_local

                step = 0.05
                for _ in range(30):
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            y = x_local.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5, 5)
                            fy = eval_func(y)
                            if fy < f_local:
                                x_local, f_local = y, fy
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 0.005:
                            break

                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()

        return best_x, best_f
