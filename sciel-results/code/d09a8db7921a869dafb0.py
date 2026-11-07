"""Optimizer for BBOB f19 (Griewank-Rosenbrock) using CMA-ES multi-start."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer for smooth low-dimensional functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        self._call_count = 0
        budget = self.budget
        dim = self.dim
        deadline = self._start_time + 270.0

        def counted_func(x):
            self._call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return float(func(x_clipped))

        def time_ok():
            return time.time() < deadline and self._call_count < budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Quick random evaluations for early AOCC progress
        n_early = min(10, budget)
        for _ in range(n_early):
            if not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, dim)
            f = counted_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES multi-start
        # CMA-ES parameters
        pop_size = max(4 + int(3 * np.log(dim)), 6)  # ~12 for dim=5
        sigma0 = 2.0  # initial step size

        # Starting points: center, axis points, random
        starts = [np.zeros(dim)]
        for i in range(dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        for _ in range(5):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        for start in unique_starts:
            if not time_ok():
                break

            # CMA-ES implementation
            mean = start.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.eye(dim)
            cs = (pop_size + 2) / (dim + 4)
            ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
            cc = (4 + dim / pop_size) / (dim + 4 + 2 * pop_size / dim)
            cmu = min(1 - 1 / ds, 2 / ((dim + 2) ** 2 + pop_size))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            max_iter = max(5, min(30, (budget - self._call_count) // pop_size))

            for it in range(max_iter):
                if not time_ok():
                    break

                # Generate population
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T))
                D_inv = 1.0 / (D + 1e-15)

                w = np.zeros(pop_size)
                mu_w = 0.0
                lam = pop_size
                # Weights: top half get more weight
                for i in range(lam // 2):
                    w[i] = np.log(lam / 2 + 0.5) - np.log(i + 1)
                w = w / w.sum()
                mu_w = np.sum(w[:int(np.sum(w > 0))]) ** 2

                x_pop = np.zeros((lam, dim))
                for i in range(lam):
                    zi = rng.standard_normal(dim)
                    x_pop[i] = mean + sigma * (B @ (D_inv * zi))
                    x_pop[i] = np.clip(x_pop[i], -5.0, 5.0)

                # Evaluate
                f_pop = np.zeros(lam)
                for i in range(lam):
                    if not time_ok():
                        break
                    f_pop[i] = counted_func(x_pop[i])

                if self._call_count >= budget or time.time() >= deadline:
                    break

                # Sort by fitness
                idx = np.argsort(f_pop)
                x_pop = x_pop[idx]
                f_pop = f_pop[idx]

                # Update best
                if f_pop[0] < best_f:
                    best_f = f_pop[0]
                    best_x = x_pop[0].copy()

                # Compute new mean
                y_w = np.zeros(dim)
                for i in range(lam // 2):
                    y_w += w[i] * x_pop[i]
                new_mean = y_w

                # Update path and covariance
                y_mean = (new_mean - mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_w) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) / chiN < (1 + 2 / (dim + 1)) * 1.4
                pc = (1 - cc) * pc + hsig * cc * (y_mean + cc * ps)
                C = (1 - cs * ds - cmu) * C + cs * ds * np.outer(ps, ps)
                for i in range(lam // 2):
                    yi = (x_pop[i] - mean) / sigma
                    C += cmu * w[i] * np.outer(yi, yi)

                mean = new_mean
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                sigma = max(sigma, 1e-10)

                # Convergence check
                if sigma < 1e-8:
                    break

            if not time_ok():
                break

        # Phase 3: Local refinement with Nelder-Mead on best point
        if time_ok() and self._call_count < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': min(50, budget - self._call_count),
                             'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Ensure at least one call was made
        if self._call_count == 0:
            counted_func(np.zeros(dim))

        return best_x, best_f
