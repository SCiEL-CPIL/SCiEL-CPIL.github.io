"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            if call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES implementation
        def cma_es(x0, sigma0, remaining_budget, max_generations=None):
            nonlocal call_count, best_x, best_f, deadline

            if remaining_budget < 2 * (4 + int(3 * np.log(dim))):
                return x0, best_f

            n = dim
            # Population size
            lam = int(np.ceil(4 + 3 * np.log(n)))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            gen = 0
            if max_generations is None:
                max_generations = (remaining_budget - 10) // lam

            for gen in range(max_generations):
                if call_count + lam > budget or time.time() > deadline:
                    break

                # Generate population
                ys = []
                xs = []
                for i in range(lam):
                    y = rng.standard_normal(n)
                    x_off = np.clip(x_mean + sigma * (B @ (D * y)), -5.0, 5.0)
                    ys.append(y)
                    xs.append(x_off)

                # Evaluate
                fs = []
                for x_off in xs:
                    if call_count >= budget or time.time() > deadline:
                        break
                    f_val = wrapped(x_off)
                    fs.append(f_val)

                if len(fs) < lam:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order]
                xs_sorted = [xs[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Update mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * xs_sorted[i]

                # Update step size
                y_mean = np.zeros(n)
                for i in range(mu):
                    y_mean += weights[i] * ys_sorted[i]

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_mean

                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0

                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])

                D = D * np.exp((c_1 / 2) * (1 - delta_h) + (c_mu / 2) * (np.trace(rank_mu) / n - 1))
                B = B @ (np.eye(n) + c_1 * (delta_h * (1 / (D ** 2)) * rank_1 - np.eye(n)) + c_mu * (rank_mu / D ** 2))

                # SVD to maintain orthonormality
                U, S, Vt = np.linalg.svd(B)
                B = U @ Vt
                D = S * D

                x_mean = x_new

                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = xs_sorted[0].copy()

                # Stagnation check
                if gen > 10 and np.std(fs_sorted[:mu]) < 1e-10:
                    break

            return x_mean, best_f

        # Phase 1: Initial diverse scout
        starts = []
        # Nominal minima candidates for Rosenbrock
        starts.append(np.ones(dim))
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))
        starts.append(np.array([1.0, 1.0, 0.5, 0.5, 0.5]))
        starts.append(np.array([0.5, 0.5, 1.0, 1.0, 1.0]))

        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate starting points
        for s in starts:
            if call_count >= budget or time.time() > deadline:
                break
            s_clipped = np.clip(s, -5.0, 5.0)
            wrapped(s_clipped)

        if best_x is None:
            best_x = np.ones(dim)
            best_f = wrapped(best_x)

        # Phase 2: Run CMA-ES from best point
        remaining = budget - call_count
        if remaining > 30 and time.time() < deadline:
            # Start with moderate sigma
            sigma0 = 1.0
            _, _ = cma_es(best_x.copy(), sigma0, remaining)

        # Phase 3: If still budget left, restart CMA-ES with smaller sigma for polish
        remaining = budget - call_count
        if remaining > 20 and time.time() < deadline and best_f > 1e-6:
            sigma0 = 0.1
            _, _ = cma_es(best_x.copy(), sigma0, remaining)

        # Phase 4: Final micro-refinement
        remaining = budget - call_count
        if remaining > 5 and time.time() < deadline and best_f > 1e-8:
            sigma0 = 0.01
            _, _ = cma_es(best_x.copy(), sigma0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
