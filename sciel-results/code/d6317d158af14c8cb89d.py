"""Black-box continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for 5-dim box-constrained problems."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(dim))  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * dim) + 1 / (24 * dim ** 2))

        # Mean update control
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial parameters
        x = np.zeros(dim)  # Start at center of box
        sigma = 2.0  # Initial step size
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # CMA-ES state
        ps = np.zeros(dim)
        hs = 1.0
        pc = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Eigenvalue decomposition for matrix square root
        def _update_matrix_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_sqrt, C_sqrt_inv

        C_sqrt, C_sqrt_inv = _update_matrix_sqrt(C)

        # Helper to clip to box
        def clip_to_box(x):
            return np.clip(x, -5, 5)

        # Main loop
        while calls < budget:
            # Check time limit
            if time.time() > deadline - 5.0:
                break

            # Generate population
            y_list = []
            f_list = []

            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = clip_to_box(y)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(y_list) < lambda_:
                # Pad if we didn't generate full population
                while len(y_list) < lambda_:
                    y_list.append(y_list[-1])
                    f_list.append(f_list[-1])

            # Sort by fitness
            indices = np.argsort(f_list)
            y_list = [y_list[i] for i in indices]
            f_list = [f_list[i] for i in indices]

            # Select best mu individuals
            y_selected = np.array(y_list[:mu])
            f_selected = np.array(f_list[:mu])

            # Weighted mean of selected
            y_mean = np.sum(weights[:, None] * y_selected, axis=0)

            # Update path for sigma control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_sqrt_inv @ ((y_mean - x) / sigma))
            hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi_lambda else 0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_lambda - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update path for covariance matrix
            pc = (1 - c_c) * pc + hs * c_c * (y_mean - x) / sigma

            # Update covariance matrix
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None, None] * np.outer(y_selected - x, y_selected - x), axis=0)
            C = (1 + c_1 * (1 - hs) - c_1 - c_mu) * C + c_1 * (rank_one - pc @ pc.T) + c_mu * rank_mu / sigma ** 2

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            C_sqrt, C_sqrt_inv = _update_matrix_sqrt(C)

            # Update mean
            x = y_mean

            # If sigma becomes very small, consider restarting
            if sigma < 1e-8:
                # Restart with a larger sigma around best point
                x = best_x.copy()
                sigma = 1.0
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt, C_sqrt_inv = _update_matrix_sqrt(C)
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
