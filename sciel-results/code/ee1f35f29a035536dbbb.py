"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and quadratic surrogate refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # Initialize near origin (minimizer known to be in [-4,4]^5)
        x = rng.uniform(-3, 3, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES state
        sigma = 2.0
        C = np.eye(d)  # Covariance matrix
        p_sigma = np.zeros(d)  # Evolution path for sigma
        p_c = np.zeros(d)  # Evolution path for C
        c_sigma = 1.0 / (d + 4)
        c_c = (2 + 2.0 / (d + 1)) / ((d + 2) + 2.0 / (d + 1))
        mu = max(1, d // 2)  # Number of best solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = (2 / ((d + 1.3) ** 2 + mu_eff)) / 5.0
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (d + 2)) / ((d + 2) + mu_eff))

        # History for potential surrogate model
        history_x = [x.copy()]
        history_f = [fx]

        calls_used = 1

        for step in range(n - 1):
            # Check time budget
            if time.time() > deadline:
                break

            # Generate population
            pop_x = np.zeros((mu, d))
            pop_f = np.zeros(mu)

            for i in range(mu):
                # Sample from multivariate normal
                z = rng.multivariate_normal(np.zeros(d), C, size=1).flatten()
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                pop_x[i] = y
                pop_f[i] = func(y)
                calls_used += 1

                # Track best
                if pop_f[i] < best_f:
                    best_x = y.copy()
                    best_f = pop_f[i]

                # Record in history
                history_x.append(y.copy())
                history_f.append(pop_f[i])

                # Check time mid-loop
                if time.time() > deadline:
                    break

            if calls_used >= n:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x_sorted = pop_x[idx]
            pop_f_sorted = pop_f[idx]

            # Update mean
            w = weights[:min(mu, len(pop_x_sorted))]
            if len(w) < mu:
                w = np.pad(w, (0, mu - len(w)), constant_values=0)
            x = np.sum(w[:, np.newaxis] * pop_x_sorted[:len(w)], axis=0)

            # Update evolution paths
            y_inv = np.linalg.cholesky(C + 1e-10 * np.eye(d))
            y_inv_inv = np.linalg.inv(y_inv.T)

            delta = (x - pop_x_sorted[0]) / (sigma + 1e-15)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv_inv @ delta
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (step + 1))) / np.sqrt(mu_eff) < (1 + 2 / (d + 1)) * np.sqrt(1 - c_sigma)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta

            # Update covariance matrix
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.sum(w[i, np.newaxis, np.newaxis] * np.outer(pop_x_sorted[i] - x, pop_x_sorted[i] - x) for i in range(len(w)))
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure C is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma (CSA)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            d_sigma = 1 + 2 * max(0, (mu_eff - 2) / (d + 2)) + c_sigma
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Keep history bounded
            if len(history_x) > 80:
                history_x = history_x[-80:]
                history_f = history_f[-80:]

            # Periodic quadratic surrogate refinement when close
            if step % 50 == 0 and len(history_x) >= 2 * d + 5 and best_f < 50:
                # Fit quadratic model to recent points
                recent_k = min(40, len(history_x))
                X = np.array(history_x[-recent_k:])
                F = np.array(history_f[-recent_k:])

                # Build feature matrix for quadratic: [1, x_i, x_i*x_j]
                n_params = 1 + d + d * (d + 1) // 2
                M = np.ones((recent_k, n_params))
                M[:, 1:d + 1] = X
                col = d + 1
                for i_idx in range(d):
                    for j_idx in range(i_idx, d):
                        M[:, col] = X[:, i_idx] * X[:, j_idx]
                        col += 1

                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(M, F, rcond=None)
                    if rank >= n_params - 2:
                        # Extract gradient and Hessian
                        c1 = coeffs[1:1 + d]
                        H = np.zeros((d, d))
                        for i in range(d):
                            for j in range(i, d):
                                idx = 1 + d + i + j
                                H[i, j] = 2.0 * coeffs[idx]
                                H[j, i] = H[i, j]

                        eigvals_H = np.linalg.eigvalsh(H)
                        if np.all(eigvals_H > 1e-8):
                            # Newton step
                            try:
                                step_vec = np.linalg.solve(H, -c1)
                                step_norm = np.linalg.norm(step_vec)
                                if step_norm > 3.0:
                                    step_vec = step_vec / step_norm * 3.0
                                x_new = np.clip(x + step_vec, -5, 5)
                                fx_new = func(x_new)
                                calls_used += 1
                                if fx_new < best_f:
                                    best_x = x_new.copy()
                                    best_f = fx_new
                                    x = x_new
                                history_x.append(x_new.copy())
                                history_f.append(fx_new)
                            except np.linalg.LinAlgError:
                                pass
                except Exception:
                    pass

                if calls_used >= n or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
