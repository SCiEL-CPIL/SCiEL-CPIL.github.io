"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset.

Combines CMA-ES (from current) with a Differential Evolution global search phase
(inspiration 1's structural idea) followed by a short (1+1)-ES polish.
The DE phase uses a moderate population and mutation to escape local optima that
CMA-ES may converge to prematurely, while CMA-ES refines the best region found.
"""
import numpy as np
import time


class Optimizer:
    """DE global search + CMA-ES refinement + (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # --- Phase 1: Differential Evolution (global exploration) ---
        de_budget = max(50, budget // 3)
        pop_size = max(10, 4 * dim + 2)  # ~22 for dim=5
        F = 0.8
        CR = 0.9

        # Initialize population
        population = np.array([rng.uniform(-4.0, 4.0, dim) for _ in range(pop_size)])
        fitness = np.array([func(clip(ind)) for ind in population])
        calls = pop_size

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # DE loop
        while calls < de_budget and (time.time() - start) < deadline:
            new_fitness = fitness.copy()
            new_population = population.copy()

            for i in range(pop_size):
                if calls >= de_budget:
                    break
                if (time.time() - start) > deadline:
                    break

                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                # Ensure they are distinct from i
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)

                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                # Crossover
                cross_point = rng.integers(0, dim)
                trial = np.where(
                    rng.random(dim) < CR or np.arange(dim) == cross_point,
                    mutant,
                    population[i]
                )
                trial = clip(trial)

                f_trial = func(trial)
                calls += 1

                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

            population = new_population
            fitness = new_fitness

        # --- Phase 2: CMA-ES refinement around best point ---
        cma_budget = max(80, budget - calls - 30)  # reserve 30 for polish
        x_mean = best_x.copy()
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        lam = 4 * dim + 2
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        while calls < cma_budget and (time.time() - start) < deadline:
            zs = rng.standard_normal((lam, dim))
            ys = np.array([x_mean + sigma * (B @ (D * z)) for z in zs])
            fs = np.array([func(clip(y)) for y in ys])
            calls += lam

            if (time.time() - start) > deadline:
                break

            # Update best
            for j in range(lam):
                if fs[j] < best_f:
                    best_f = fs[j]
                    best_x = ys[j].copy()

            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            x_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * ys[:mu], axis=0)
            y = (x_mean - x_old) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * np.sqrt(dim):
                pc = (1 - c_c) * pc + c_c * y
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (np.outer(pc, pc) + (c_c - c_c ** 2) * np.eye(dim))
                     + c_mu * np.sum(w[:mu, None, None] *
                                    np.outer((ys[:mu] - x_old) / sigma, (ys[:mu] - x_old) / sigma),
                                    axis=0))
                ev_vals, ev_vecs = np.linalg.eigh(C)
                ev_vals = np.abs(ev_vals)
                idx = ev_vals.argsort()[::-1]
                ev_vals = ev_vals[idx]
                ev_vecs = ev_vecs[:, idx]
                D = np.sqrt(ev_vals)
                B = ev_vecs

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = min(max(sigma, 1e-10), 50.0)

        # --- Phase 3: (1+1)-ES polish ---
        x = best_x.copy()
        fx = best_f
        s = 0.1
        polish_budget = budget - calls

        while calls < budget and (time.time() - start) < deadline:
            y = clip(x + s * rng.standard_normal(dim))
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                s *= 1.15
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                s *= 0.85
            if s < 1e-8:
                break

        return best_x, best_f
