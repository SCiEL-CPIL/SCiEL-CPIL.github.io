"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive sigma, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_run(self, func, x0, sigma0, remaining_budget, rng, deadline):
        """Run CMA-ES from x0 with sigma0, using up to remaining_budget evaluations."""
        n = self.dim
        pop_size = int(4 + 3 * np.log(n))  # For n=5, pop_size = 4 + 3*ln(5) ≈ 4+4.83=8
        pop_size = max(4, pop_size)

        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 1.3) + 4 + n / (n + 4))
        cs = (n + 1.5 + cc) / (n + 5.5 + cc)
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()
        c1 = (2 / ((n + 1.8) ** 2 + mueff)) * (2 - cc) / 2
        cmu = min(1 - c1, (2 * (mueff - 2 / 3) / ((n + 2) ** 2 + mueff)) * (2 - cc) / 2)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        x = np.array(x0, dtype=float)
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        while calls < remaining_budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            Y = []
            for i in range(pop_size):
                zi = rng.standard_normal(n)
                yi = x + sigma * (B @ (D * zi))
                Y.append(yi)
            Y = np.array(Y)
            calls += pop_size

            # Evaluate
            F = np.array([func(y) for y in Y])

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = Y[idx_min].copy()

            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Mean update
            y_w = sum(weights[i] * Y[i] for i in range(mu))
            x_old = x.copy()
            x = x + y_w

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D ** -1 @ (y_w / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - pop_size)))) < (1.4 + 2 / (n + 1)) * chi_n
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Update C matrix
            if hsig:
                pc = (1 - cc) * pc + cc / np.sqrt(cc * (2 - cc) * mueff) * (y_w / sigma)
            else:
                pc = (1 - cc) * pc

            delta_sig = (1 - hsig) * cc * (2 - cc) / (n + 1)
            C = (1 - c1 - cmu + delta_sig) * C + c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(Y[i] - x_old, Y[i] - x_old)

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Eigen decomposition
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-20))
            B = B2
            D = D2

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, use 290s

        rng = np.random.default_rng(42)
        n = self.dim

        # Initial diverse points
        initial_points = []
        # Center
        initial_points.append(np.zeros(n))
        # Random points
        for _ in range(9):
            initial_points.append(rng.uniform(-5, 5, n))
        # Axis-aligned points
        for axis in range(n):
            for val in [-3, 3]:
                p = np.zeros(n)
                p[axis] = val
                initial_points.append(p)

        # Evaluate initial points
        best_x = None
        best_f = np.inf
        calls = 0
        for p in initial_points:
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            fp = func(p)
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Run CMA-ES from the best point
        if best_x is not None:
            remaining = self.budget - calls
            if remaining > 0:
                bx, bf, used = self._cmaes_run(func, best_x, 1.0, remaining, rng, deadline)
                if bf < best_f:
                    best_f = bf
                    best_x = bx
                calls += used

        # If we have time and budget left, do more restarts from different points
        while calls < self.budget:
            if time.time() > deadline:
                break
            # Try a new random center
            x0 = rng.uniform(-5, 5, n)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = self.budget - calls
            if remaining < 10:
                break

            # Short CMA-ES run
            bx, bf, used = self._cmaes_run(func, x0, 0.5, min(remaining, 50), rng, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx
            calls += used

        return best_x, best_f
# EVOLVE-BLOCK-END
