"""Combined DE + CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE for global search + CMA-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_cmaes(x0, sigma0, max_calls):
            """Run CMA-ES with scipy minimize."""
            nonlocal calls_made, x_best, f_best
            local_x = np.array(x0, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass
            return local_f

        def de_search(max_calls, pop_size=12, F_mut=0.8, CR=0.5):
            """Differential evolution for global exploration."""
            nonlocal calls_made, x_best, f_best
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            pop_size = min(pop_size, max(8, max_calls // 15))

            # Initialize population
            pop = []
            for _ in range(pop_size):
                x = rng.uniform(-5, 5, dim)
                f = evaluate(x)
                pop.append((x, f))

            pop.sort(key=lambda item: item[1])

            # If we already have a good point, include it
            if x_best is not None:
                pop[0] = (x_best.copy(), f_best)
                pop.sort(key=lambda item: item[1])

            while calls_made < max_calls and time.time() < deadline:
                x_best_pop = pop[0][0]
                new_pop = []

                for i in range(pop_size):
                    if calls_made >= max_calls or time.time() > deadline:
                        new_pop.append(pop[i])
                        continue

                    indices = [j for j in range(pop_size) if j != i]
                    if len(indices) < 3:
                        new_pop.append(pop[i])
                        continue
                    r1, r2, r3 = rng.choice(indices, 3, replace=False)

                    # Mutation: best/1/bin
                    v = x_best_pop + F_mut * (pop[r1][0] - pop[r2][0])

                    # Crossover
                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i][0][j]

                    trial = np.clip(trial, -5, 5)
                    f_trial = evaluate(trial)

                    if f_trial <= pop[i][1]:
                        new_pop.append((trial, f_trial))
                    else:
                        new_pop.append(pop[i])

                pop = new_pop
                pop.sort(key=lambda item: item[1])

            if pop and pop[0][1] < f_best:
                f_best = pop[0][1]
                x_best = pop[0][0].copy()

        def local_refine(x_center, max_calls):
            """Local refinement around a point using Nelder-Mead."""
            nonlocal calls_made, x_best, f_best
            if x_center is None or max_calls < 5 or time.time() > deadline:
                return

            local_x = np.array(x_center, dtype=np.float64)
            local_f = f_best

            def wrapped(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return local_f
                x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
                fx = float(func(x))
                calls_made += 1
                if fx < local_f:
                    local_f = fx
                    x_best = x.copy()
                return fx

            try:
                res = minimize(wrapped, local_x, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'maxfev': max_calls,
                                        'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < f_best:
                    f_best = float(res.fun)
                    x_best = np.array(res.x, dtype=np.float64)
            except Exception:
                pass

        def random_restart_search(max_calls):
            """Random restarts with local refinement."""
            nonlocal calls_made, x_best, f_best
            rng = np.random.default_rng(int(time.time() * 2000) % (2**31))
            remaining = max_calls
            while remaining >= 10 and time.time() < deadline:
                x = rng.uniform(-4, 4, dim)
                f = evaluate(x)
                remaining -= 1
                if f < f_best:
                    f_best = f
                    x_best = x.copy()
                # Quick local refinement
                if remaining >= 15 and time.time() < deadline:
                    local_refine(x_best.copy(), min(15, remaining))
                    remaining = max_calls - (calls_made - (max_calls - remaining))
                    remaining = min(remaining, max_calls - calls_made)

        # Strategy:
        # 1. DE for global exploration (majority of budget)
        # 2. Local refinement on best points found
        # 3. Additional DE or restarts if budget remains

        # Phase 1: DE for global exploration
        de_calls = int(budget * 0.6)
        if time.time() < deadline:
            de_search(de_calls, pop_size=15)

        # Phase 2: Local refinement on best point
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            refine_calls = min(remaining, 50)
            local_refine(x_best.copy(), refine_calls)

        # Phase 3: More DE if budget remains
        remaining = budget - calls_made
        if remaining >= 20 and time.time() < deadline:
            de_calls2 = min(remaining, int(remaining * 0.7))
            de_search(de_calls2, pop_size=10)

        # Phase 4: Final local refinement
        remaining = budget - calls_made
        if remaining >= 10 and x_best is not None and time.time() < deadline:
            refine_calls = min(remaining, 30)
            local_refine(x_best.copy(), refine_calls)

        # Phase 5: Random restarts if still budget left
        remaining = budget - calls_made
        if remaining >= 15 and time.time() < deadline:
            random_restart_search(remaining)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
