import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, sigma0, max_calls):
            nonlocal calls, best_x, best_f
            if max_calls < 20:
                return

            # CMA-ES parameters
            n = self.dim
            # lambda population size
            lam = max(4 + int(3 * n), 8)
            # step size
            sigma = sigma0
            # mean
            m = x0.copy()
            # covariance matrix
            C = np.eye(n)
            # evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # weights
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            # step size adaptation
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            # covariance matrix adaptation
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # chi_n
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # B and D for C = B B^T
            B = np.eye(n)
            D = np.ones(n)

            # history of step sizes
            sigma_history = []

            while calls < budget and time.time() < deadline - 5:
                # sample population
                Y = np.random.randn(lam, n)
                # transform to search space
                X = m + sigma * (Y @ (B * D).T)
                # evaluate
                F = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    f = try_eval(X[i])
                    if f is None:
                        break
                    F.append(f)
                if len(F) < lam:
                    break

                # sort by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = np.array(F)[order]

                # update mean
                m_new = m + sigma * sum(weights[i] * Y_sorted[i] for i in range(mu))
                m = m_new

                # update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (
                    B @ (D * (Y_sorted[:mu].T @ weights)))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # update covariance
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam))) < (1.4 + 2 / (n + 1)) * chi_n:
                    v = (1 - c_sigma) * (1 - c_c) * np.outer(pc, pc)
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (m - m_new) / sigma
                    pc = np.clip(pc, -10, 10)
                    C = (1 + c_1 * (np.outer(pc, pc) - C) + c_mu * sum(
                        weights[i] ** 2 * np.outer(Y_sorted[i], Y_sorted[i]) for i in range(mu)) - c_1 - c_mu) * C
                    # eigendecomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    B = eigvecs
                    D = np.sqrt(eigvals)

                # convergence check
                if np.linalg.norm(m - m_new) < 1e-8 * sigma or sigma < 1e-10:
                    break

        # Phase 1: Sobol quasi-random sampling for global coverage
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=n, scramble=True)
        num_sobol = 100
        points = sampler.random(num_sobol)
        points = 10.0 * points - 5.0  # map to [-5, 5]

        # Evaluate Sobol points
        for i in range(num_sobol):
            if calls >= 120 or time.time() > deadline - 10:
                break
            try_eval(points[i])

        # Add center and a few random points
        if calls < 130:
            try_eval(np.zeros(n))
        for _ in range(10):
            if calls >= 130 or time.time() > deadline - 10:
                break
            try_eval(np.random.uniform(-5, 5, n))

        # Phase 2: CMA-ES from the best point found
        if best_x is not None and calls < budget - 30:
            # Estimate initial sigma from the spread of good points
            # Use the distance to the second-best point as a heuristic
            cma_es(best_x.copy(), sigma0=1.0, max_calls=budget - calls)

        return best_x, best_f
