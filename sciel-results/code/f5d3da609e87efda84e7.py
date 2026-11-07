"""CMA-ES based optimizer for BBOB f8 (Rosenbrock) with multi-start and adaptive restarts"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start exploration, adaptive restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        total_calls = 0

        def check_time():
            return time.time() < deadline

        dim = self.dim

        # Phase 1: Multi-point exploration to find a good starting point
        # Rosenbrock has a narrow valley; random sampling helps locate it
        n_init = min(20, self.budget // 10)
        for _ in range(n_init):
            if total_calls >= self.budget or not check_time():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if total_calls >= self.budget or not check_time():
            return best_x, best_fx

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # ~8 for dim=5
        sigma = 1.0
        mean = best_x.copy()

        # CMA-ES state
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        c_sigma = (lam / 2 + 1) / (dim + 2 + lam / 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Restart mechanism
        restart_count = 0
        max_restarts = 5
        no_improve_count = 0
        max_no_improve = 30

        while total_calls < self.budget and check_time():
            # Check for stagnation
            old_best_fx = best_fx

            # Generate offspring
            ys = []
            fs = []

            for i in range(lam):
                if total_calls >= self.budget or not check_time():
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (Csqrt @ z)
                y = np.clip(y, -5, 5)

                fy = func(y)
                total_calls += 1
                ys.append(y)
                fs.append(fy)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(ys) < lam:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update mean
            y_w = np.sum(weights[:, None] * ys[:mu], axis=0)
            mean = y_w

            # Update sigma
            y_sigma = (y_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (total_calls // lam + 1))) / chi_n < 1.4 + 2 / (dim + 1)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

            # Update covariance matrix
            if h_sigma:
                rank_one = np.outer(p_c, p_c)
            else:
                rank_one = np.zeros((dim, dim))

            rank_mu = np.sum(weights[:, None, None] * (ys[:mu] - mean)[:, None, :] * (ys[:mu] - mean)[:, :, None], axis=0)

            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Cholesky decomposition
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                Csqrt = np.eye(dim)
                mean = best_x.copy()
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                continue

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Prevent sigma from becoming too small or too large
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

            # Stagnation detection
            if best_fx > old_best_fx - 1e-10:
                no_improve_count += 1
            else:
                no_improve_count = 0

            # Adaptive restart: if stuck, restart from best with larger sigma
            if no_improve_count >= max_no_improve:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                mean = best_x.copy()
                sigma = max(1.0, sigma * 2.0)
                C = np.eye(dim)
                Csqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                max_no_improve = max(15, max_no_improve // 2)

        return best_x, best_fx
# EVOLVE-BLOCK-END
