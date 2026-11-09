"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard for BBOB f5 (linear slope)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize at center
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Step size
        sigma = 3.0

        # CMA-ES matrices
        C = np.eye(n)
        Cs = np.zeros((n, n))
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((lam - 1) / n) - 1)) / (n + 3)
        c_c = (4 + lam / n) / (n + 4 + 2 * lam / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lam)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / ((n + 2) * (lam - mu_eff + 0.5)) - 1))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Number of iterations
        max_iters = budget - 1
        call_count = 1

        rng = np.random.default_rng(42)

        for it in range(max_iters):
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Generate population
            y = rng.standard_normal((lam, n))

            # Sort by distance from mean
            dists = np.linalg.norm(y, axis=1)
            order = np.argsort(dists)

            # Update mean
            y_w = np.sum(weights[:, None] * y[order[:mu]], axis=0)
            x_new = np.clip(x + sigma * y_w, -5, 5)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + c_sigma * chi_n * y_w
            hsig = (1 - np.sqrt((1 - c_sigma) ** (2 * (it + 1))) /
                    (1 - c_sigma) ** (2 * (it + 1))) < (1 + 2 * np.sqrt((mu_eff + 1) / (n + 1))) * c_sigma
            if hsig:
                pc = (1 - c_c) * pc + hsig * c_c * chi_n * y_w / sigma
            else:
                pc = (1 - c_c) * pc

            # Update covariance
            delta_hsig = hsig * (1 - c_c) - 1
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig * np.trace(C)) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y[order[:mu]], y[order[:mu]]), axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Generate and evaluate population
            for i in range(lam):
                if time.time() > deadline - 1.0:
                    break
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n) * np.sqrt(np.diag(C).max())

                z = rng.standard_normal(n)
                y_trial = x_new + sigma * (L @ z)
                y_trial = np.clip(y_trial, -5, 5)

                call_count += 1
                fy = func(y_trial)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y_trial.copy()

            x = x_new

        # Final polish: local search around best point
        if time.time() < deadline - 5.0:
            # Fine-tune with small steps
            step = 0.01
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        trial = best_x.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5, 5)
                        call_count += 1
                        ft = func(trial)
                        if ft < best_fx:
                            best_x = trial
                            best_fx = ft
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
