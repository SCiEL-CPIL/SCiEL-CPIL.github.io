import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        def time_up():
            return time.time() >= deadline

        # Phase 1: Random sampling to find a good starting point
        n_initial = min(20, budget // 10)
        best_x = None
        best_fx = np.inf
        for i in range(n_initial):
            if time_up():
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls = n_initial
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls = 1

        # Phase 2: CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        x = best_x.copy()
        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

        gen = 0
        while calls + lam <= budget and not time_up():
            # Generate population
            population = []
            for i in range(lam):
                if time_up():
                    break
                y = B @ (D * rng.standard_normal(dim))
                xi = np.clip(x + sigma * y, -5, 5)
                fi = func(xi)
                calls += 1
                population.append((fi, xi, y))

            if not population:
                break

            population.sort(key=lambda t: t[0])

            # Update best
            if population[0][0] < best_fx:
                best_fx = population[0][0]
                best_x = population[0][1].copy()

            # Mean update
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i][1]
            x_new = np.clip(x_new, -5, 5)

            # CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * B @ (D * (x_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / sigma

            # Covariance update
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(x_new - x, x_new - x) + delta_hsig * np.outer(pc, pc))
            for i in range(mu):
                C += cmu * weights[i] * np.outer(population[i][1] - x, population[i][1] - x)

            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            B = B.T

            x = x_new
            sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / chi_n - 1))
            gen += 1

        # Phase 3: Local refinement with remaining budget
        remaining = budget - calls
        if remaining > 0 and not time_up():
            x_local = best_x.copy()
            fx_local = best_fx
            step = 0.5
            for _ in range(min(remaining, 60)):
                if time_up():
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if time_up():
                            break
                        x_trial = x_local.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < fx_local:
                            x_local = x_trial
                            fx_local = f_trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            if fx_local < best_fx:
                best_x, best_fx = x_local, fx_local

        return best_x, best_fx
