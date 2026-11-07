"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES for BBOB f12 (Bent Cigar) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # --- CMA-ES parameters ---
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2                # number of recombination weights
        # Standard weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Step-size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial point: try several random starts, keep best
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        for _ in range(10):
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        x = best_x.copy()
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        remaining = budget - 10  # account for initial evaluations
        if remaining < 1:
            return best_x, best_f

        for gen in range(remaining):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, dim))
            for i in range(lambda_):
                yi = B @ (D * rng.standard_normal(dim))
                ys[i] = x + sigma * yi

            # Evaluate population
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                fs[i] = func(ys[i])
                if time.time() > deadline:
                    # Fill remaining with inf and break
                    fs[i + 1:] = np.inf
                    break

            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best solution
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Weighted recombination
            x_new = np.dot(weights[:mu], ys[:mu])

            # Update covariance matrix
            y_weighted = (x_new - x) / sigma
            p_c = (1 - c_c) * p_c + c_c * y_weighted
            hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1))
                    / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma)

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (
                hsig / c_sigma * y_weighted)

            C = B @ np.diag(D ** 2) @ B.T
            C = ((1 + c_1 * (2 - c_1 - c_mu) + c_mu) * C
                 - c_1 * np.outer(p_c, p_c)
                 + c_mu / c_sigma ** 2 * np.outer(p_sigma, p_sigma))
            C = 0.5 * (C + C.T)

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = x_new
            sigma *= np.exp(c_sigma / d_sigma *
                            (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Clip x to box
            x = np.clip(x, -5, 5)

        # --- Polish phase with Nelder-Mead if time allows ---
        if time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead on the current best
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10,
                             'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
