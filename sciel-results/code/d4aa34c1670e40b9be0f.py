"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the shifted/rotated Rosenbrock task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES parameters
        lambda_ = max(4 * n, 2 * n)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1.0 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_m = min(1 - c_c, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Initial point: center of box, slight randomization
        x = rng.uniform(-0.5, 0.5, n)
        x = clip_box(x)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        C = np.eye(n)
        C_inv = np.eye(n)
        # Eigendecomposition for sqrt
        D, V = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        C_sqrt = V @ np.diag(D) @ V.T
        C_sqrt_inv = V @ np.diag(1.0 / D) @ V.T

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        count = 1

        # Adaptive restart: if stagnation detected, restart with new sigma
        stagnation_counter = 0
        prev_fx = fx

        while count < budget:
            # Sample lambda_ candidates
            z = rng.standard_normal((lambda_, n))
            y = x + sigma * (C_sqrt @ z.T).T
            y = clip_box(y)
            f_y = np.array([func(yp) for yp in y])

            # Sort by fitness
            order = np.argsort(f_y)
            y_sorted = y[order]
            f_sorted = f_y[order]

            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Check stagnation
            if abs(f_sorted[0] - prev_fx) < 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_fx = f_sorted[0]

            # Restart if long stagnation
            if stagnation_counter > 50:
                # Reset to best found, reduce sigma
                x = best_x.copy()
                fx = best_fx
                sigma *= 0.5
                C = np.eye(n)
                C_inv = np.eye(n)
                D, V = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                C_sqrt = V @ np.diag(D) @ V.T
                C_sqrt_inv = V @ np.diag(1.0 / D) @ V.T
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                stagnation_counter = 0
                count += 1
                if count >= budget:
                    break
                continue

            # Weighted mean of top mu
            m = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Success history for sigma
            delta_vec = C_sqrt_inv @ (m - x)
            h_sigma = 1 if (np.sum(p_sigma ** 2) / (1 - (1 - c_sigma) ** (2 * count)) <
                            (1 + 2 / (mu_eff + 1)) * chi_n ** 2) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m - x)
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 50.0)

            # Success history for evolution path
            h_sigma_path = 1 if (np.sum(p_sigma ** 2) / (1 - (1 - c_sigma) ** (2 * count)) <
                                 (1 + 2 / (mu_eff + 1)) * chi_n ** 2) else 0
            p_c = (1 - c_c) * p_c + h_sigma_path * np.sqrt(
                c_c * (2 - c_c) * mu_eff) * delta_vec

            # Update covariance
            delta_C = (1 / c_m) * (
                p_c @ p_c - C
            )
            delta_C += (1 - (1 - c_m) ** 2) * (
                np.sum(weights * ((y_sorted[:mu] - x) ** 2), axis=0) - np.diag(C)
            )
            C += c_m * (delta_C + (1 - c_m) * (np.sum(weights, axis=0) *
                np.sum(weights * (y_sorted[:mu] - m) @ (y_sorted[:mu] - m).T, axis=0) - np.diag(C)))

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                D, V = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C = V @ np.diag(D) @ V.T
                C = (C + C.T) / 2
                D_inv = 1.0 / D
                C_inv = V @ np.diag(D_inv) @ V.T
                C_sqrt = V @ np.diag(np.sqrt(D)) @ V.T
                C_sqrt_inv = V @ np.diag(1.0 / np.sqrt(D)) @ V.T
            except:
                C = np.eye(n)
                C_inv = np.eye(n)
                C_sqrt = np.eye(n)
                C_sqrt_inv = np.eye(n)

            x = m
            fx = f_sorted[0]
            count += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
