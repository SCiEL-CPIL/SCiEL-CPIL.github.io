import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # 30s safety margin

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Initial point: center of box
        x_mean = np.zeros(dim)
        sigma = 2.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES constants
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))

        # Cholesky factor
        B = np.eye(dim)

        def update_cholesky():
            nonlocal B
            try:
                B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                B = np.eye(dim)

        best_x = None
        best_f = np.inf
        calls = 0

        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = x_mean + sigma * (B @ z)
                y = np.clip(y, -5.0, 5.0)
                f = func(y)
                calls += 1
                offspring.append((f, y))
                if f < best_f:
                    best_f = f
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # Sort and select
            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu]
            y_w = np.array([p[1] for p in selected])

            # New mean
            x_new = np.sum(w[:, None] * y_w, axis=0)

            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C)
            for i in range(mu):
                zi = (y_w[i] - x_mean) / sigma
                C += c_mu * w[i] * np.outer(zi, zi)
            C = (C + C.T) / 2

            # Regularize if needed
            try:
                evals, evecs = np.linalg.eigh(C)
                if np.any(evals < 1e-14):
                    evals = np.maximum(evals, 1e-14)
                    C = (evecs * evals) @ evecs.T
                    C = (C + C.T) / 2
            except:
                pass

            # Update pc
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (x_new - x_mean) / sigma)

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            x_mean = x_new
            update_cholesky()

        # Polish phase: use remaining budget with (1+1)-ES from best point
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_p = max(sigma * 0.1, 1e-3)
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma_p * np.random.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_p *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_p *= 0.9

        return best_x, best_f
