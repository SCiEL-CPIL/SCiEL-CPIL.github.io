import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: coarse LHS bracketing then single CMA-ES with coordinate polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def cma_es(x0, sigma0, max_iter=200):
            nonlocal calls_made, best_x, best_f
            n = dim
            pop_size = int(4 + 3 * np.log(n))
            pop_size = max(pop_size, n + 1)
            mu = pop_size // 2

            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / ((n + 1) ** 2 + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 - mu_eff / (n + 1)) / ((n + 2) ** 2 + mu_eff)
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            L = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            population = np.zeros((pop_size, n))
            fvals = np.zeros(pop_size)

            for i in range(pop_size):
                if not time_ok():
                    break
                z = np.random.randn(n)
                y = L @ z
                x = mean + sigma * y
                fvals[i] = safe_call(x)
                population[i] = x

            for it in range(max_iter):
                if not time_ok():
                    break

                order = np.argsort(fvals)
                population = population[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = population[0].copy()

                if np.std(fvals[:mu]) < 1e-12 * (1 + abs(fvals[0])):
                    break

                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (population[i] - mean) / sigma

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                mean = mean + c_c * (y_w / c_sigma - p_sigma)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

                C = (1 + c_1 + c_mu - c_1 / n) * C
                for i in range(mu):
                    y_i = (population[i] - mean) / sigma
                    C -= c_1 * np.outer(y_i, y_i)
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals2, eigvecs2 = np.linalg.eigh(C)
                    eigvals2 = np.clip(eigvals2, 1e-12, None)
                    L = eigvecs2 @ np.diag(np.sqrt(eigvals2))

                for i in range(pop_size):
                    if not time_ok():
                        break
                    z = np.random.randn(n)
                    y = L @ z
                    x = mean + sigma * y
                    fvals[i] = safe_call(x)
                    population[i] = x

            return best_x, best_f

        def coordinate_polish(x_start, max_rounds=100):
            nonlocal calls_made, best_x, best_f
            x = x_start.copy()
            f_cur = safe_call(x)
            if not np.isfinite(f_cur):
                return x, f_cur
            
            step = 0.5
            for _ in range(max_rounds):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = safe_call(x_new)
                        if np.isfinite(f_new) and f_new < f_cur - 1e-10:
                            x = x_new
                            f_cur = f_new
                            improved = True
                            break
                    if not time_ok():
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            return x, f_cur

        # Phase 1: Coarse 3-level LHS for rapid bracketing (3^5 = 243 points, use 25)
        n_levels = 3
        n_samples = 25
        rng = np.random.RandomState(42)
        lhs_points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            # Map to 3 levels: -3.33, 0, 3.33 approximately
            level_indices = (np.arange(n_samples) + rng.rand(n_samples)) / n_samples * n_levels
            level_indices = np.clip(level_indices.astype(int), 0, n_levels - 1)
            lhs_points[perm, d] = (level_indices + 0.5) / n_levels * 10 - 5

        for p in lhs_points:
            if not time_ok():
                break
            safe_call(p)

        # Phase 2: Single high-precision CMA-ES from best point
        if time_ok() and best_x is not None:
            x, f = cma_es(best_x, sigma0=0.25, max_iter=200)
            best_x, best_f = x, f

        # Phase 3: Coordinate descent polishing
        if time_ok() and best_x is not None and calls_made < budget - 10:
            x, f = coordinate_polish(best_x, max_rounds=50)
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
