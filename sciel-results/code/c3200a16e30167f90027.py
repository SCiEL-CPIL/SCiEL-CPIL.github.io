"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def count_call(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_x = x.copy()
                best_f = f
            return f

        def cmaes(start_x, sigma, max_calls):
            nonlocal calls, best_x, best_f
            n = dim
            pop = int(4 + 3 * np.log(n))
            # Initialize
            x_mean = start_x.copy()
            f_mean = count_call(x_mean, func(x_mean)) if calls < budget else np.inf
            if calls >= budget:
                return

            # CMA-ES parameters
            c_sigma = (n + 2) / (4 * n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + pop / n) / (4 + n + pop / n)
            c_1 = 2 / ((n + 1.3) ** 2 + pop)
            c_mu = min(1 - c_1, 2 * ((pop / 2 + 1) / (n + 2) ** 2 - 1))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            hsig = 1.4 + 2 / (n + 1)

            # Cumulation
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_eigval = np.ones(n)
            C_eigvec = np.eye(n)
            # Precompute
            step_delta = np.zeros(n)
            step_delta_sq = 0.0
            step_delta_cross = np.zeros((n, n))

            for gen in range(max_calls):
                if time.time() > deadline or calls >= budget:
                    break

                # Sample
                B = C_eigvec
                D = np.sqrt(C_eigval)
                B_inv = np.linalg.inv(B)
                ys = []
                xs = []
                fs = []
                for i in range(pop):
                    if time.time() > deadline or calls >= budget:
                        break
                    z = np.random.randn(n)
                    y = B @ (D * z)
                    x_new = np.clip(x_mean + sigma * y, -5, 5)
                    f_new = func(x_new)
                    count_call(x_new, f_new)
                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_new)

                if len(xs) < pop:
                    break

                # Sort
                order = np.argsort(fs)
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                fs = [fs[i] for i in order]

                # Weights
                mu = pop // 2
                w = np.zeros(mu)
                for i in range(mu):
                    w[i] = np.log(mu + 1) - np.log(i + 1)
                w = w / w.sum()
                mueff = 1 / np.sum(w ** 2)

                # Update
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * ys[i]

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (B_inv @ y_w)
                norm_ps = np.linalg.norm(p_sigma)
                if norm_ps / (1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 1e10)

                # Update C
                p_c = (1 - c_c) * p_c + c_c * ((norm_ps / chi_n < hsig) * y_w - (1 - c_c) * p_c)
                hsig = 1.4 + 2 / (n + 1)

                # Rank-1 update
                delta = np.outer(p_c, p_c) / n
                delta -= np.eye(n)

                # Rank-mu update
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += np.outer(ys[i], ys[i])
                rank_mu *= -c_mu / mueff

                C = (1 + c_1 * (delta + rank_mu)) * C

                # Eigen decomposition
                try:
                    C_eigval, C_eigvec = np.linalg.eigh(C)
                    C_eigval = np.clip(C_eigval, 1e-12, None)
                except:
                    break

                # Update mean
                x_mean = np.clip(np.sum([w[i] * xs[i] for i in range(mu)], axis=0), -5, 5)

                # Check termination
                if np.all(np.abs(C_eigval) < 1e-12):
                    break

            return

        # Multiple restarts
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            if r == 0:
                # First restart: explore broadly
                start_x = np.random.uniform(-5, 5, dim)
                sigma = 2.0
            else:
                # Subsequent restarts: start near best found
                if best_x is not None:
                    start_x = best_x + np.random.uniform(-1, 1, dim)
                    start_x = np.clip(start_x, -5, 5)
                    sigma = 0.5
                else:
                    start_x = np.random.uniform(-5, 5, dim)
                    sigma = 2.0

            cmaes(start_x, sigma, calls_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
