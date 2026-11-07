"""CMA-ES approach for BBOB f7 (Step Ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f7.
    CMA-ES adapts its search distribution to the problem's geometry,
    handling rotations, shifts, and scale changes without gradients.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        call_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def _counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return np.inf
            if time.time() > deadline - 10:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters for dim=5
        lambda_ = int(4 + 3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2  # number of best points used for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial setup
        mean = np.zeros(dim)
        sigma = 2.0  # larger initial step size to explore [-5,5]^5
        C = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Chi distribution approximation for expected norm of standard normal
        chi_mu = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        def _update_chol(C_mat):
            """Update Cholesky decomposition of C."""
            try:
                L = np.linalg.cholesky(C_mat)
                return L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-10)
                C_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C_reg)
                return L

        # CMA-ES main loop
        for iteration in range(1000):
            if call_count >= budget - 5 or time.time() > deadline - 15:
                break

            # Update Cholesky factor
            C_sqrt = _update_chol(C)
            C_sqrt_inv = np.linalg.inv(C_sqrt)

            # Generate lambda_ candidates
            z = np.random.randn(lambda_, dim)
            y = np.array([C_sqrt @ zi for zi in z])
            x_candidates = mean + sigma * y

            # Evaluate all candidates
            f_values = np.array([_counted_func(xi) for xi in x_candidates])

            # Sort by fitness
            idx = np.argsort(f_values)
            x_sorted = x_candidates[idx]
            f_sorted = f_values[idx]
            y_sorted = y[idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Recombination: weighted average of best mu points
            w = weights[:mu]
            mean_old = mean.copy()
            mean = mean + sigma * np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (mean - mean_old) / sigma)

            # Check for delta rule (sigma adaptation)
            ps_norm = np.linalg.norm(ps)
            if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 1 / dim) * chi_mu:
                sigma = sigma * np.exp(c_sigma / d_sigma * (ps_norm / chi_mu - 1))

            # Update evolution path for C
            y_mean = np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance matrix
            delta_hs = (1 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 1 / dim) * chi_mu else 0)
            C = (1 + c_1 * (delta_hs - 1) - c_1 - c_mu) * C + \
                c_1 * (np.outer(pc, pc) + delta_hs - 1) * np.eye(dim) + \
                c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu, :], y_sorted[:mu, :]), axis=0)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        # Final local refinement with perturbations around the best point
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 20:
            rng = np.random.default_rng(42)
            for i in range(min(remaining // 3, 10)):
                if call_count >= budget - 5 or time.time() > deadline - 10:
                    break
                for step in [0.1, 0.5, 1.0]:
                    if call_count >= budget - 5:
                        break
                    perturbation = rng.normal(0, step, size=dim)
                    x_try = np.clip(best_x + perturbation, -5, 5)
                    f_try = _counted_func(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        # Ensure valid return
        if np.isnan(best_f) or np.isinf(best_f):
            best_f = 0.0
            best_x = np.zeros(dim)

        return best_x, best_f

# EVOLVE-BLOCK-END
