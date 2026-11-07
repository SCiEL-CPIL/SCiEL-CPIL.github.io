"""
Population-based Differential Evolution (DE/rand/1) with adaptive mutation factor
and final Nelder-Mead polish for BBOB-style black-box optimization.

Strategy:
- Phase 1: Latin hypercube sampling for diverse initialization.
- Phase 2: DE/rand/1 with adaptive F (mutation factor) and CR (crossover rate).
  The population's implicit covariance helps navigate multimodal landscapes.
- Phase 3: Nelder-Mead polish on the best point if budget remains.
"""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(np.clip(func(x), -1e15, 1e15))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Latin Hypercube Sampling for initialization
        pop_size = max(10, min(40, budget // 10))
        n_init = min(pop_size, budget // 4)

        # Latin hypercube in [-5, 5]^dim
        if n_init > 0:
            lhs_points = np.zeros((n_init, dim))
            for d in range(dim):
                # Permute bins
                bins = np.linspace(-5, 5, n_init + 1)
                indices = rng.permutation(n_init)
                for i in range(n_init):
                    low = bins[indices[i]]
                    high = bins[indices[i] + 1]
                    lhs_points[i, d] = rng.uniform(low, high)
            # Ensure center is included
            lhs_points[0] = np.zeros(dim)

            for i in range(n_init):
                if calls[0] >= budget or time.time() > deadline:
                    break
                x = lhs_points[i]
                f_val = wrapped(x)
                update_best(x, f_val)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 2: Differential Evolution
        # Initialize population around best point with some diversity
        pop = np.zeros((pop_size, dim))
        pop_f = np.zeros(pop_size)

        # Keep the best point in population
        pop[0] = best_x.copy()
        pop_f[0] = best_f

        # Fill rest with points near best + random perturbation
        for i in range(1, pop_size):
            if calls[0] >= budget or time.time() > deadline:
                break
            # Sample around best with decreasing radius
            radius = 2.0 * np.sqrt(1.0 / (i + 1))
            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            x = best_x + radius * direction
            x = np.clip(x, -5, 5)
            f_val = wrapped(x)
            pop[i] = x
            pop_f[i] = f_val
            update_best(x, f_val)

        # Adaptive parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        stagnation_count = 0
        max_stagnation = 15

        for gen in range(2000):
            if calls[0] >= budget - 5 or time.time() > deadline:
                break

            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            pop[0] = best_x.copy()
            pop_f[0] = best_f

            # Check stagnation
            if pop_f[0] >= best_f - 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Adapt F and CR based on stagnation
            if stagnation_count > max_stagnation:
                # Restart with more exploration
                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.5, 1.0)
                # Perturb population
                for i in range(1, pop_size):
                    if calls[0] >= budget - 5 or time.time() > deadline:
                        break
                    radius = rng.uniform(1.0, 3.0)
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    x = best_x + radius * direction
                    x = np.clip(x, -5, 5)
                    f_val = wrapped(x)
                    pop[i] = x
                    pop_f[i] = f_val
                    update_best(x, f_val)
                stagnation_count = 0
                continue

            # DE/rand/1: for each individual, create trial vector
            for i in range(pop_size):
                if calls[0] >= budget - 5 or time.time() > deadline:
                    break

                # Select three distinct random individuals
                while True:
                    a, b, c = rng.choice(pop_size, size=3, replace=False)
                    if len(set([a, b, c, i])) == 4:
                        break

                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])

                # Crossover
                j_rand = rng.integers(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i, j]

                # Clip to bounds
                trial = np.clip(trial, -5, 5)

                # Evaluate
                f_trial = wrapped(trial)

                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    update_best(trial, f_trial)

                # Slowly decrease F for exploitation
                F = max(0.3, F * 0.999)

        # Phase 3: Nelder-Mead polish
        if best_x is not None and calls[0] < budget - 5 and time.time() < deadline:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Final L-BFGS-B polish if budget allows
        if best_x is not None and calls[0] < budget - 3 and time.time() < deadline:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
