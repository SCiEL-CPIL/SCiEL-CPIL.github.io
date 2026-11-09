"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast CMA-ES with multi-restart for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def elapsed():
            return time.time() - start_time

        def remaining():
            return budget - calls_made

        def time_ok():
            return elapsed() < deadline - 5

        # Phase 1: Random sampling to find a good basin
        n_random = min(30, budget // 15)
        for i in range(n_random):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: CMA-ES with fast convergence
        # Use a smaller population for speed, but enough for multimodal landscape
        x = best_x.copy()
        sigma = 1.5
        lambda_ = 8  # smaller population for speed
        mu = 4
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        c1 = 1 / (dim + 3)
        cmu = 2 / ((dim + 2) ** 2 + 5)
        cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        step_size = sigma

        # Precompute Cholesky factor for sampling
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        for gen in range(2000):
            if calls_made >= budget or not time_ok():
                break

            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget or not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = L @ z
                xi = x + step_size * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                offspring.append((fi, xi))

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            for fi, xi in offspring:
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Take top mu
            mu_actual = min(mu, len(offspring))
            selected = [offspring[i][1] for i in range(mu_actual)]
            selected_f = [offspring[i][0] for i in range(mu_actual)]

            # Compute mean of selected
            new_x = np.zeros(dim)
            for i in range(mu_actual):
                new_x += weights[i] * selected[i]

            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
            step_size *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            step_size = np.clip(step_size, 1e-10, 50)

            # Update covariance
            v = (new_x - x) / step_size
            rank_one = c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T)
            rank_mu = cmu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) / step_size ** 2 for i in range(mu_actual))
            C = (1 - c1 - cmu) * C + rank_one + rank_mu

            # Update pc
            pc = (1 - cc) * pc + cc * hsig * v

            x = new_x

            # Ensure C is symmetric positive definite and update Cholesky
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

            # If converged, do a local restart
            if step_size < 1e-6:
                # Try a small perturbation
                for _ in range(3):
                    if calls_made >= budget or not time_ok():
                        break
                    x_try = x + rng.normal(0, 0.5, dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        x = x_try.copy()
                        break
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
