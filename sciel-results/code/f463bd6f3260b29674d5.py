"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Differential Evolution for global search,
    Nelder-Mead for local refinement, with quadratic model guidance."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Initial diverse sampling to get a rough landscape
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-4, 4, size=dim)
            evaluate(x)

        if global_best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)

        # Phase 2: Differential Evolution for global exploration
        # Use a moderate population size and run for a portion of the budget
        pop_size = 12
        de_calls_budget = min(budget - calls_made, 250)
        if de_calls_budget < 20:
            de_calls_budget = max(0, budget - calls_made)

        # Initialize DE population around best and random points
        pop = []
        # Seed with best
        pop.append(global_best_x.copy())
        # Add diverse points
        for i in range(pop_size - 1):
            if calls_made >= budget or not time_ok():
                break
            # Mix: some near best, some random
            if i < pop_size // 3:
                x = global_best_x + 1.5 * rng.standard_normal(dim)
            else:
                x = rng.uniform(-5, 5, size=dim)
            x = np.clip(x, -5, 5)
            fx = evaluate(x)
            if fx is not None:
                pop.append(x)

        # Ensure we have enough population members
        while len(pop) < pop_size and calls_made < budget and time_ok():
            x = rng.uniform(-5, 5, size=dim)
            fx = evaluate(x)
            if fx is not None:
                pop.append(x)

        # DE parameters
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        best_in_pop_idx = 0
        best_in_pop_fx = global_best_fx

        # Re-evaluate to find best in pop
        pop_fx = []
        for i, p in enumerate(pop):
            if i == 0:
                pop_fx.append(global_best_fx)
            else:
                # We already have these values from evaluation, but let's just track
                # Actually, we need to be careful. Let's just re-evaluate if needed
                # For efficiency, assume we tracked them. But our evaluate doesn't return a list.
                # Let's just find the best in pop by re-checking logic or storing.
                # Since we don't store fx per pop member easily, let's just use global best for now
                # and let DE find improvements.
                # Better approach: store fx for pop members.
                pass

        # Let's restart the DE logic slightly to properly track fitness
        pop = []
        pop_fx = []
        # Re-initialize population properly
        x_best = global_best_x.copy()
        fx_best = global_best_fx
        pop.append(x_best)
        pop_fx.append(fx_best)

        for i in range(pop_size - 1):
            if calls_made >= budget or not time_ok():
                break
            if i < pop_size // 3:
                x = global_best_x + 1.5 * rng.standard_normal(dim)
            else:
                x = rng.uniform(-5, 5, size=dim)
            x = np.clip(x, -5, 5)
            fx = evaluate(x)
            if fx is not None:
                pop.append(x)
                pop_fx.append(fx)

        # Run DE
        max_de_iters = de_calls_budget // pop_size
        for it in range(max_de_iters):
            if calls_made >= budget or not time_ok():
                break

            # Sort population by fitness
            indices = np.argsort(pop_fx)
            pop = [pop[i] for i in indices]
            pop_fx = [pop_fx[i] for i in indices]

            # Update global best
            if pop_fx[0] < global_best_fx:
                global_best_fx = pop_fx[0]
                global_best_x = pop[0].copy()

            new_pop = []
            new_pop_fx = []

            # Keep the best individual (elitism)
            new_pop.append(pop[0].copy())
            new_pop_fx.append(pop_fx[0])

            for i in range(1, pop_size):
                if calls_made >= budget or not time_ok():
                    break

                # Select 3 distinct random individuals different from i
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)

                # Create trial vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                ft = evaluate(trial)
                if ft is None:
                    # Out of budget, keep original
                    new_pop.append(pop[i].copy())
                    new_pop_fx.append(pop_fx[i])
                    continue

                # Selection
                if ft <= pop_fx[i]:
                    new_pop.append(trial)
                    new_pop_fx.append(ft)
                else:
                    new_pop.append(pop[i].copy())
                    new_pop_fx.append(pop_fx[i])

            if len(new_pop) == pop_size:
                pop = new_pop
                pop_fx = new_pop_fx
            else:
                # If we ran out of budget or time, break
                break

        # Phase 3: Local refinement with Nelder-Mead
        if global_best_x is not None and calls_made < budget and time_ok():
            from scipy.optimize import minimize

            x0 = global_best_x.copy()
            remaining_calls = budget - calls_made
            # Limit Nelder-Mead iterations to not exceed budget
            # NM uses roughly (n+1) evaluations per iteration
            max_nm_iters = max(1, remaining_calls // (dim + 1))

            # Use a wrapper to count calls and stop if budget exceeded
            nm_calls = [0]

            def nm_func(x):
                nonlocal calls_made, global_best_x, global_best_fx
                if calls_made >= budget or not time_ok():
                    return np.inf
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                nm_calls[0] += 1
                if fx < global_best_fx:
                    global_best_fx = fx
                    global_best_x = x.copy()
                return fx

            try:
                res = minimize(nm_func, x0, method='Nelder-Mead',
                               options={'maxiter': max_nm_iters,
                                        'xatol': 1e-6, 'fatol': 1e-8,
                                        'adaptive': True})
            except Exception:
                pass

        # Phase 4: Final local search with coordinate descent if budget remains
        if global_best_x is not None and calls_made < budget and time_ok():
            x = global_best_x.copy()
            fx = global_best_fx
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]

            for step in step_sizes:
                if calls_made >= budget or not time_ok():
                    break
                improved = True
                while improved:
                    improved = False
                    for j in range(dim):
                        if calls_made >= budget or not time_ok():
                            break
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[j] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            fx_trial = evaluate(x_trial)
                            if fx_trial is not None and fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                improved = True
                                break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
