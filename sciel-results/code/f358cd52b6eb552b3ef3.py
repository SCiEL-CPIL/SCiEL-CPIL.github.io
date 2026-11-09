"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # 10s safety margin before 300s limit

        # Best solution found so far
        best_x = np.full(dim, 0.0)
        best_f = np.inf
        calls_made = 0

        def count_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # CMA-ES parameters
        lambda_ = 4 * dim + 4  # population size
        # We'll use a simplified CMA-ES with rank-mu update

        def cma_es(x0, f0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            n = dim
            lam = lambda_
            mu = lam // 2  # number of parents
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)
            # Step size control
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
            # Covariance matrix adaptation
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = (2 - c_c + mueff / n) / ((n + 2) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # Initial values
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            x_mean = x0.copy()
            f_mean = f0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            def inv_sqrt_mat(M):
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-20)
                return eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

            C_inv_sqrt = inv_sqrt_mat(C)

            for gen in range(max_calls // lam + 1):
                if calls_made >= max_calls or time.time() > deadline:
                    break

                # Generate offspring
                offsets = rng.standard_normal((lam, n))
                ys = [x_mean + sigma * (C_inv_sqrt @ z) for z in offsets]
                # Clip to bounds
                ys = [np.clip(y, -5, 5) for y in ys]
                # Evaluate
                fs = []
                for y in ys:
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    fy = count_call(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(fs) < lam:
                    break

                # Sort
                idx = np.argsort(fs)
                ys_sorted = [ys[i] for i in idx]
                fs_sorted = [fs[i] for i in idx]

                # Update mean
                x_mean_new = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)

                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50)

                # Update covariance
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean_new - x_mean) / sigma

                delta_s = 1 + (2 * (np.linalg.norm(ps) / chi_n < (1.4 + 1 / n) * chi_n) - 1) * c_c
                C = (1 + c_1 + c_mu - c_c * c_c) * C
                C += c_1 * (pc @ pc.T - C)
                rank_mu = sum(w[i] * np.outer(ys_sorted[i] - x_mean, ys_sorted[i] - x_mean) for i in range(mu))
                C += c_mu * rank_mu
                C = (C + C.T) / 2

                x_mean = x_mean_new
                C_inv_sqrt = inv_sqrt_mat(C)

            return best_x, best_f

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # First, do a coarse grid / random search to find a good starting point
        # Use 50 random points to initialize
        n_init = 50
        init_xs = [np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(n_init)]
        init_fs = []
        for x in init_xs:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = count_call(x)
            init_fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort and pick top few as starting points for CMA-ES
        if init_fs:
            idx = np.argsort(init_fs)
            top_k = min(3, len(idx))
            starting_points = [(init_xs[i], init_fs[i]) for i in idx[:top_k]]
        else:
            starting_points = [(np.zeros(dim), np.inf)]

        # Allocate remaining budget among CMA-ES runs
        remaining = budget - calls_made
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        n_cma = min(3, max(1, remaining // 100))
        per_run = remaining // n_cma

        for i in range(n_cma):
            if calls_made >= budget or time.time() > deadline:
                break
            if i < len(starting_points):
                x0, f0 = starting_points[i]
            else:
                x0 = best_x.copy()
                f0 = best_f
            # Use best found as starting point for later runs
            if i > 0:
                x0 = best_x.copy()
                f0 = best_f

            sigma0 = max(0.5, 2.0 / (i + 1))
            cma_es(x0, f0, sigma0, per_run)

        # Final polish: local search with very small steps
        if calls_made < budget and time.time() < deadline:
            # Nelder-Mead-like local refinement
            x = best_x.copy()
            f = best_f
            step = 0.01
            for _ in range(min(20, budget - calls_made)):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = count_call(y)
                        if fy < f:
                            x, f = y, fy
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
