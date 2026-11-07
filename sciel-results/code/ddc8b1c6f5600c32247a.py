"""CMA-ES optimizer for BBOB f12 (Bent cigar) with quadratic surrogate polish."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initialize near origin (minimizer known to be in [-4,4]^5)
        best_x = rng.uniform(-2, 2, dim)
        best_f = func(best_x)
        calls = 1

        # CMA-ES state
        mean = best_x.copy()
        sigma = 2.0
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for covariance
        ps = np.zeros(dim)  # Evolution path for step size
        c_c = (4 + dim / 3.0) / ((dim + 4) * 3 + 4)
        c_s = (2 + 2.0 / (dim + 1.0)) / ((dim + 1) + 2 + 2.0 / (dim + 1.0))
        mu = max(1, dim // 2)  # Number of selected parents
        lamb = 2 * mu  # Population size
        weights = np.zeros(lamb)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_cov = (2 + 2.0 / (dim + 1.0) - mu_eff / ((dim + 1) + 2.0 / (dim + 1.0))) / (
            (dim + 1.3) ** 2 + mu_eff
        )
        d_orth = 1.0 + 2.0 * max(0, (mu_eff - 2) / (dim - 2)) - c_cov
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # History for quadratic model
        history_x = [best_x.copy()]
        history_f = [best_f]

        # Phase 1: CMA-ES exploration (use ~60% of budget)
        cma_budget = int(budget * 0.6)
        for step in range(cma_budget):
            if calls >= budget or time.time() > deadline:
                break

            # Generate population
            pops = []
            fvals = []
            for i in range(lamb):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform using C and sigma
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                    y = mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback if C is not PD
                    y = mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pops.append(y)
                fvals.append(fy)

                # Update best
                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy

                # Keep history bounded
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > 50:
                    history_x.pop(0)
                    history_f.pop(0)

            # Sort by fitness
            sorted_idx = np.argsort(fvals)
            sorted_pops = [pops[i] for i in sorted_idx[:mu]]
            sorted_fvals = [fvals[i] for i in sorted_idx[:mu]]

            # Update mean
            new_mean = np.sum(np.array([weights[i] * sorted_pops[i] for i in range(mu)]), axis=0)

            # Update evolution paths
            diff = (new_mean - mean) / sigma
            ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * diff
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_s) ** (2 * (step + 1))) < (
                (1.4 + 2.0 / (dim + 1)) * chi_n
            ):
                pc = (1 - c_c) * pc + np.sqrt(
                    c_c * (2 - c_c) * mu_eff
                ) * (diff - (1 - c_s) * ps)
            else:
                pc = (1 - c_c) * pc

            # Update covariance
            delta_h = (
                np.linalg.norm(ps) ** 2 / chi_n**2
                - 1
                * (1 - (1 - c_s) ** (2 * (step + 1)))
            )
            C = (
                (1 + c_cov * (mu_eff - 2) - c_cov) * C
                + c_cov
                * (
                    np.outer(pc, pc) / c_c
                    + (delta_h - 1 + c_c * (2 - c_c)) * np.outer(pc, pc)
                    + np.sum(
                        [
                            weights[i] * np.outer(sorted_pops[i] - mean, sorted_pops[i] - mean)
                            for i in range(mu)
                        ],
                        axis=0,
                    )
                    / sigma**2
                )
            )

            # Ensure C is symmetric and PD
            C = (C + C.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            sigma *= np.exp(
                c_s / chi_n * (np.linalg.norm(ps) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-6, 50.0)

            mean = new_mean

        # Phase 2: Quadratic surrogate model refinement
        polish_budget = budget - calls
        for i in range(polish_budget):
            if calls >= budget or time.time() > deadline:
                break

            # Build quadratic model from recent history
            n_hist = min(25, len(history_x))
            if n_hist >= dim + 1:
                X = np.array(history_x[-n_hist:])
                F = np.array(history_f[-n_hist:])

                # Fit quadratic: f(x) ≈ c + b^T x + x^T A x
                d = dim
                n_params = 1 + d + d * (d + 1) // 2

                # Build feature matrix
                M = np.ones((n_hist, n_params))
                M[:, 1 : d + 1] = X
                col = d + 1
                for i_idx in range(d):
                    for j_idx in range(i_idx, d):
                        M[:, col] = X[:, i_idx] * X[:, j_idx]
                        col += 1

                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(M, F, rcond=None)
                    if rank >= n_params - 1:
                        # Extract Hessian from quadratic coefficients
                        H = np.zeros((d, d))
                        col = d + 1
                        for i_idx in range(d):
                            for j_idx in range(i_idx, d):
                                H[i_idx, j_idx] = coeffs[col]
                                if i_idx != j_idx:
                                    H[j_idx, i_idx] = coeffs[col]
                                else:
                                    H[i_idx, i_idx] = 2 * coeffs[col]
                                col += 1

                        # Gradient at best_x
                        grad = coeffs[1 : d + 1].copy()
                        for i_idx in range(d):
                            for j_idx in range(d):
                                if i_idx != j_idx:
                                    grad[i_idx] += H[i_idx, j_idx] * best_x[j_idx]
                            grad[i_idx] += H[i_idx, i_idx] * best_x[i_idx]

                        # Check if Hessian is positive definite enough
                        eigvals = np.linalg.eigvalsh(H)
                        if np.min(eigvals) > 1e-8:
                            # Newton step
                            try:
                                step_vec = np.linalg.solve(H, grad)
                                step_norm = np.linalg.norm(step_vec)
                                if step_norm > sigma:
                                    step_vec = step_vec * sigma / step_norm
                                y = best_x - step_vec
                                y = np.clip(y, -5, 5)
                            except np.linalg.LinAlgError:
                                y = best_x + sigma * rng.standard_normal(dim)
                                y = np.clip(y, -5, 5)
                        else:
                            # Fallback: use gradient descent step
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-12:
                                y = best_x - 0.5 * grad / grad_norm * sigma
                                y = np.clip(y, -5, 5)
                            else:
                                y = best_x + sigma * rng.standard_normal(dim)
                                y = np.clip(y, -5, 5)
                    else:
                        y = best_x + sigma * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                except Exception:
                    y = best_x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
            else:
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

            fy = func(y)
            calls += 1

            history_x.append(y.copy())
            history_f.append(fy)

            if fy < best_f:
                best_x = y.copy()
                best_f = fy
                sigma *= 0.8
                sigma = max(sigma, 0.001)
            else:
                sigma *= 0.9
                sigma = max(sigma, 0.001)

            if len(history_x) > 50:
                history_x.pop(0)
                history_f.pop(0)

        return best_x, best_f
