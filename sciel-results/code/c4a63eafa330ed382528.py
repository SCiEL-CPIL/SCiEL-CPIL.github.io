"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal Natural Gradient Descent for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 275

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting ----
        lambda_ = 14
        mu_ = 7
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # ---- Diagonal step-size adaptation constants ----
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ---- state ----
        x_mean = np.zeros(dim)
        sigma = np.full(dim, 2.0)  # per-dimension step sizes
        p_sigma = np.zeros(dim)

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

        # ---- initial diverse exploration ----
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

        # ---- main loop: diagonal natural gradient descent ----
        while call_count < budget:
            if timed_out():
                break

            # generate population
            offspring = []
            for _ in range(lambda_):
                if call_count >= budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * z
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

            # update per-dimension step sizes
            # p_sigma tracks the weighted mean of z-vectors
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

            # adapt sigma per dimension
            norm_p = np.linalg.norm(p_sigma)
            if norm_p > 1e-15:
                # scale each dimension independently
                sigma *= np.exp(c_sigma / d_sigma * (norm_p / chi_n - 1))

            # per-dimension adaptation based on progress
            # if a dimension has small relative change, shrink its step size
            rel_change = np.abs(delta) / (sigma + 1e-15)
            for i in range(dim):
                if rel_change[i] < 0.1:
                    sigma[i] *= 0.95
                elif rel_change[i] > 1.0:
                    sigma[i] *= 1.05

            sigma = np.clip(sigma, 1e-10, 15.0)

            # stagnation guard
            if np.all(sigma < 1e-7):
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                sigma = np.full(dim, 0.5)

        # ---- final polish with Nelder-Mead ----
        if call_count < budget and best_x is not None and not timed_out():
            from scipy.optimize import minimize
            remaining = budget - call_count
            if remaining > 10:
                def polish_func(x):
                    nonlocal call_count
                    if call_count >= budget or timed_out():
                        return np.inf
                    y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    f = func(y)
                    call_count += 1
                    if f < best_f:
                        best_f = float(f)
                        best_x = y.copy()
                    return f

                try:
                    res = minimize(
                        polish_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 200),
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
