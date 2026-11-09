"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast CMA-ES (direct Cholesky update) for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 270  # safety margin vs 300s / 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting (moderate for dim=5) ----
        lambda_ = 14
        mu_ = 7
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # ---- CMA-ES strategy constants ----
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ---- state ----
        x_mean = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        L = np.eye(dim)          # Cholesky: C = L L^T
        have_L = True

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(y)
            call_count += 1
            if f < best_f:
                best_f = float(f)
                best_x = y.copy()
            return f

        # ---- initial diverse exploration: break the Discus ring symmetry ----
        n_init = min(16, budget)
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            if k == 0:
                x0 = np.zeros(dim)
            elif k == 1:
                x0 = 3.0 * rng.standard_normal(dim)
            elif k == 2:
                x0 = np.linspace(-4.0, 4.0, dim)
            else:
                x0 = 4.0 * rng.uniform(-1.0, 1.0, dim)
            evaluate(x0)

        # ---- main CMA-ES loop ----
        while call_count < budget:
            if timed_out():
                break

            # generate population
            offspring = []
            for _ in range(lambda_):
                if call_count >= budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                f = evaluate(y)
                offspring.append((f, y, z))

            if len(offspring) < mu_:
                break

            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu_]

            # update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
            delta = x_mean - old_mean

            # evolution paths (need C^{-1/2} for p_c)
            if have_L:
                try:
                    Cinv_half_delta = np.linalg.solve(L.T, np.linalg.solve(L, delta))
                except np.linalg.LinAlgError:
                    have_L = False
                    Cinv_half_delta = delta
            else:
                Cinv_half_delta = delta

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * Cinv_half_delta

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 20.0))

            # direct covariance update
            drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
            C = (1 - c_1 - c_mu) * (L @ L.T)
            C += c_1 * (drift * np.outer(p_c, p_c) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim))
            for i in range(mu_):
                d = selected[i][1] - x_mean
                C += c_mu * w[i] * np.outer(d, d)
            C = 0.5 * (C + C.T)

            # refresh Cholesky, with PSD guard + drift detection
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                have_L = True
            except np.linalg.LinAlgError:
                L = L + 1e-4 * np.eye(dim)
                have_L = False

            # stagnation guard: if step size is negligible, re-seed mean
            if sigma < 1e-7:
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                sigma = 0.5

        # ---- final polish: local Nelder-Mead around best point if budget remains ----
        if call_count < budget and not timed_out() and best_x is not None:
            # Use a simple coordinate descent / Nelder-Mead style refinement
            # We'll do a few rounds of small perturbations around best_x
            remaining = budget - call_count
            # Allocate up to 50 calls for polish
            n_polish = min(remaining, 50)
            if n_polish > 0:
                # Use a small simplex-like search
                scale = max(sigma * 0.5, 1e-4)
                for i in range(n_polish):
                    if timed_out() or call_count >= budget:
                        break
                    # Try moving in each coordinate direction
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    for step_scale in [scale, scale * 0.1]:
                        if call_count >= budget or timed_out():
                            break
                        candidate = best_x + step_scale * direction
                        candidate = np.clip(candidate, -5.0, 5.0)
                        f = evaluate(candidate)
                        if f >= best_f:
                            # Try opposite direction
                            candidate2 = best_x - step_scale * direction
                            candidate2 = np.clip(candidate2, -5.0, 5.0)
                            f2 = evaluate(candidate2)
                            if f2 < best_f:
                                best_f = f2
                                best_x = candidate2.copy()
                            break
                        else:
                            best_f = f
                            best_x = candidate.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
