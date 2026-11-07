"""CMA-ES portfolio optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    CMA-ES portfolio: runs 3 independent CMA-ES instances with different
    starting points and population sizes to handle hidden shifts/rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, pop_size, budget, rng, deadline,
                best_x, best_f):
        """Run a single CMA-ES instance, updating best_x/best_f."""
        dim = self.dim
        n = dim
        lam = pop_size
        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((1 + 0.5 / n) / (n + 1)) - 1)) / (n + 3)
        c_c = (4 + n / 4) / (n + 4 + 2 * n / 4)
        c1 = 2 / ((n + 1.3) ** 2 + 5)
        cmu = min(1 - c1, 0.5 * (2 - (n + 2) / (n + 1.3) ** 2 - 5))
        d_sigma = 1 + 2 * max(0, np.sqrt((c_c + 1) / (n + 1)) - 1) + c1
        alpha_sigma = (1 + 2 * max(0, np.sqrt((c_c + 1) / (n + 1)) - 1)) * c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # State
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        # Precompute
        inv_sqrt_C = np.eye(n)
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Initial evaluation
        fx = func(x)
        calls += 1
        update_best(x, fx)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (inv_sqrt_C @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
                calls += 1

            if not ys:
                break

            # Evaluate
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort
            order = np.argsort(fs)
            ys_sorted = np.array(ys)[order]
            fs_sorted = np.array(fs)[order]

            # Select mu best
            mu = lam // 2
            mu_eff = 1.0 / sum(1.0 / (i + 1) for i in range(mu))
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()

            # Weighted mean
            x_new = sum(weights[i] * ys_sorted[i] for i in range(mu))

            # Update covariance
            diff = (x_new - x) / sigma
            for i in range(mu):
                d_i = (ys_sorted[i] - x) / sigma
                C += c1 * np.outer(d_i, d_i)
                C += cmu * weights[i] * np.outer(d_i, d_i)

            # Path updates
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)
            pc = (1 - c_c) * pc + c_c * diff

            # Sigma update
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp((alpha_sigma / d_sigma) * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100.0)

            # Update x
            x = x_new

            # Periodically recompute inverse square root of C
            # (do it every 10 generations to save time)
            if calls % (lam * 10) < lam:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    inv_sqrt_C = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except np.linalg.LinAlgError:
                    inv_sqrt_C = np.eye(n)

        return calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # 3 CMA-ES instances with different configs
        # Instance 1: origin, moderate pop
        # Instance 2: random point, larger pop
        # Instance 3: another random point, smaller pop
        configs = [
            {
                'x0': np.zeros(self.dim),
                'sigma0': 2.0,
                'pop': 12,
                'budget': 200
            },
            {
                'x0': rng.uniform(-4.0, 4.0, self.dim),
                'sigma0': 2.5,
                'pop': 10,
                'budget': 150
            },
            {
                'x0': rng.uniform(-4.0, 4.0, self.dim),
                'sigma0': 1.5,
                'pop': 8,
                'budget': 150
            }
        ]

        total_calls = 0
        for cfg in configs:
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break

            used = self._cma_es(
                func, cfg['x0'], cfg['sigma0'], cfg['pop'],
                cfg['budget'], rng, deadline, best_x, best_f
            )
            total_calls += used

        # Final polish: a few (1+1) steps around best_x
        if best_x is not None and total_calls < self.budget:
            remaining = self.budget - total_calls
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
