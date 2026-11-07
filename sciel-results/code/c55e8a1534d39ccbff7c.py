"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim + 1))
        lam = max(4, int(4 + 3 * np.log(dim + 1)))
        # For dim=5: lam = 4 + floor(3*1.609) = 4 + 4 = 8
        mu = lam // 2  # number of parents

        # Initial step size
        sigma = 2.0

        # Initial mean
        m = np.zeros(dim)

        # Initial covariance matrix
        C = np.eye(dim)

        # Evolution path for sigma adaptation
        p_sigma = np.zeros(dim)
        # Evolution path for covariance adaptation
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 3.5)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (dim + 2) - 1)) / ((dim + 1.3) ** 2 + mu)

        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Cholesky factor of C
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(dim)
        detC = 1.0

        calls_made = 0
        f_best = np.inf
        x_best = np.zeros(dim)

        # Track best
        def update_best(x, f):
            nonlocal f_best, x_best, calls_made
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Evaluate initial mean
        f_m = func(m)
        calls_made += 1
        update_best(m, f_m)

        generation = 0
        max_generations = 500  # safety cap

        while calls_made + lam <= budget and generation < max_generations and time.time() < deadline:
            generation += 1

            # Generate offspring
            ys = np.zeros((lam, dim))
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B @ z
                x = clip(m + sigma * y)
                ys[i] = y
                xs[i] = x
                fs[i] = func(x)
                calls_made += 1
                update_best(x, fs[i])

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update mean
            m_new = m + sigma * sum(weights[i] * ys_sorted[i] for i in range(mu))
            m_new = clip(m_new)

            # Update sigma evolution path
            # Need inverse Cholesky of C
            try:
                C_inv_sqrt = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Weighted mean of selected directions
            y_w = sum(weights[i] * ys_sorted[i] for i in range(mu))

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w

            # Hessian norm of p_sigma
            try:
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
                        / (np.sqrt(mu_eff) * np.trace(C)))
            except Exception:
                hsig = 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance evolution path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))

            delta_hsig = (1 + hsig) * (2 - c_c) - 1

            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite and update Cholesky
            try:
                # Symmetrize
                C = (C + C.T) / 2
                B = np.linalg.cholesky(C)
                detC = np.linalg.det(C)
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-6 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    B = np.linalg.cholesky(C)
                    detC = np.linalg.det(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
                    detC = 1.0

            m = m_new

            # Stopping criterion: condition number too high
            try:
                eigvals = np.linalg.eigvalsh(C)
                cond = eigvals[-1] / max(eigvals[0], 1e-15)
                if cond > 1e12:
                    break
            except Exception:
                pass

            # Stopping criterion: sigma too small
            if sigma < 1e-12:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
