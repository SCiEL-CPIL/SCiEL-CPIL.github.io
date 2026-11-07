"""Differential evolution (global) + Nelder-Mead (local) optimizer for f12."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290
        d = self.dim
        budget = self.budget
        calls = 0

        def _elapsed_ok():
            return (time.time() - start) < deadline and calls < budget

        # ---- Phase 1: Differential Evolution ----
        pop_size = 10
        F = 0.8
        CR = 0.9

        def _randn():
            nonlocal calls
            if calls >= budget or (time.time() - start) >= deadline:
                return np.zeros(d)
            calls += 1
            return func(np.random.randn(d) * 2.0)

        # initial population
        pop = np.random.uniform(-5, 5, (pop_size, d))
        fvals = np.array([func(pop[i]) for i in range(pop_size)])
        calls += pop_size

        best_i = int(np.argmin(fvals))
        best_x = pop[best_i].copy()
        best_f = fvals[best_i]

        max_gen = 200
        no_improve = 0
        for gen in range(max_gen):
            if calls + pop_size > budget or (time.time() - start) >= deadline:
                break
            trial = np.zeros_like(pop)
            new_f = np.empty(pop_size)
            for i in range(pop_size):
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                mask = np.random.rand(d) < CR
                if not mask.any():
                    j0 = np.random.randint(d)
                    mask[j0] = True
                trial[i] = np.where(mask, v, pop[i])
                trial[i] = np.clip(trial[i], -5, 5)
                new_f[i] = func(trial[i])
                calls += 1
            for i in range(pop_size):
                if new_f[i] <= fvals[i]:
                    pop[i] = trial[i]
                    fvals[i] = new_f[i]
            gen_best_i = int(np.argmin(fvals))
            if fvals[gen_best_i] < best_f - 1e-12:
                best_f = fvals[gen_best_i]
                best_x = pop[gen_best_i].copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 15:
                # mild restart: keep best, re-seed others
                idx = [k for k in range(pop_size) if k != best_i]
                for k in idx:
                    pop[k] = np.clip(best_x + np.random.randn(d) * 0.8, -5, 5)
                for k in idx:
                    fvals[k] = func(pop[k])
                    calls += 1
                no_improve = 0
                if calls >= budget:
                    break

        # ---- Phase 2: Nelder-Mead local refinement ----
        remaining = budget - calls
        if remaining >= 12:
            try:
                simplex = [best_x.copy()]
                f_simplex = [best_f]
                for j in range(d):
                    p = best_x.copy()
                    step = 0.25 if abs(best_x[j]) > 1e-8 else 0.25
                    p[j] += step
                    p = np.clip(p, -5, 5)
                    simplex.append(p)
                    f_simplex.append(func(p))
                    calls += 1

                alpha, gamma, rho, sigma_n = 1.0, 2.0, 0.5, 0.5
                nm_iter = 0
                max_nm_iter = max(1, remaining - d - 5)
                while nm_iter < max_nm_iter and calls + 1 <= budget and (time.time() - start) < deadline:
                    # order
                    order = np.argsort(f_simplex)
                    simplex = [simplex[k] for k in order]
                    f_simplex = [f_simplex[k] for k in order]
                    worst = d
                    centroid = np.mean([simplex[i] for i in range(worst)], axis=0)

                    # reflection
                    xr = np.clip(centroid + alpha * (centroid - simplex[worst]), -5, 5)
                    fr = func(xr)
                    calls += 1
                    nm_iter += 1
                    if fr < f_simplex[0]:
                        # expansion
                        xe = np.clip(centroid + gamma * (centroid - simplex[worst]), -5, 5)
                        fe = func(xe)
                        calls += 1
                        nm_iter += 1
                        if fe < fr:
                            simplex[worst] = xe
                            f_simplex[worst] = fe
                        else:
                            simplex[worst] = xr
                            f_simplex[worst] = fr
                    elif fr < f_simplex[worst - 1]:
                        simplex[worst] = xr
                        f_simplex[worst] = fr
                    else:
                        # contraction
                        if fr < f_simplex[worst]:
                            xc = np.clip(centroid + 0.5 * (xr - centroid), -5, 5)
                        else:
                            xc = np.clip(centroid - 0.5 * (simplex[worst] - centroid), -5, 5)
                        fc = func(xc)
                        calls += 1
                        nm_iter += 1
                        if fc < min(fr, f_simplex[worst]):
                            simplex[worst] = xc
                            f_simplex[worst] = fc
                        else:
                            # shrink toward best
                            for i in range(1, worst + 1):
                                simplex[i] = np.clip(simplex[0] + sigma_n * (simplex[i] - simplex[0]), -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                                nm_iter += 1
                                if calls >= budget:
                                    break
                            if calls >= budget:
                                break
                    bi = int(np.argmin(f_simplex))
                    if f_simplex[bi] < best_f:
                        best_f = f_simplex[bi]
                        best_x = simplex[bi].copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
