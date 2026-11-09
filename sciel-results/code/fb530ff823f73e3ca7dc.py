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
        deadline = t_start + 275  # safety margin vs 300s / 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting (tighter for faster convergence in dim=5) ----
        lambda_ = 12
        mu_ = 6
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
        # Discus f11 has a ring of minima at a fixed radius; a structured
        # radius sweep along a fixed direction is far more effective than
        # random points at locating the ring.
        n_init = min(20, budget)
        # Fixed direction for the radius sweep (deterministic, breaks symmetry)
        sweep_dir = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        radii = np.concatenate([
            np.linspace(0.0, 5.0, 10),       # sweep from center outward
            np.linspace(5.0, 0.0, 3),        # a few inward
            np.array([1.5, 2.5, 3.5])        # key radii
        ])
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            r = radii[k] if k < len(radii) else rng.uniform(0.0, 5.0)
            # Add a small random perturbation to avoid exact collinearity
            jitter = 0.1 * rng.standard_normal(dim)
            x0 = r * sweep_dir + jitter
            # Clip to box
            x0 = np.clip(x0, -5.0, 5.0)
            evaluate(x0)

        # ---- main CMA-ES loop ----
        while call_count < budget:
            if timed_out():
                break
            # Reserve budget for final polish
            if call_count >= budget - 30:
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

        # ---- final high-precision polish pass ----
        # Use remaining budget for a tight local search around best_x
        if best_x is not None and call_count < budget and not timed_out():
            polish_budget = min(25, budget - call_count)
            # Nelder-Mead style local refinement
            from scipy.optimize import minimize as sp_minimize
            # Use a wrapper that clips and counts calls
            polish_calls = [0]
            def polish_func(x):
                if polish_calls[0] >= polish_budget or timed_out():
                    return best_f
                y = np.clip(x, -5.0, 5.0)
                f = func(y)
                polish_calls[0] += 1
                if f < best_f:
                    best_f = float(f)
                    best_x = y.copy()
                return f
            try:
                res = sp_minimize(
                    polish_func, best_x,
                    method='Nelder-Mead',
                    options={'maxiter': polish_budget, 'xatol': 1e-10, 'fatol': 1e-12}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
