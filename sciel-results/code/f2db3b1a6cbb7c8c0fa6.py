"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus) with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        total_budget = self.budget

        # CMA-ES parameters
        # For dim=5, population size of 4 is efficient
        N = 4
        mu = N // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        c_c = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        c_1 = 2 / ((N + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))

        chi_N = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        best_x = None
        best_f = np.inf
        calls = 0

        def check_time():
            return time.time() > deadline

        # Multi-start strategy: try several initial points
        # For Discus, the minimum is often off-center, so we try a few starts
        # But we need to be careful with budget

        # Initial exploration: sample a few points to get a rough idea
        rng = np.random.default_rng(42)

        # Try a small set of initial points
        initial_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            rng.uniform(-3, 3, size=dim),
        ]

        # Evaluate initial points to find a good starting point
        for pt in initial_points:
            if calls >= total_budget or check_time():
                break
            pt = np.clip(pt, -5, 5)
            fx = func(pt)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = pt.copy()

        if best_x is None or calls >= total_budget or check_time():
            return best_x, best_f

        # Now run CMA-ES from the best initial point
        x_mean = best_x.copy()
        sigma = 1.0  # Start with moderate step size

        # Covariance matrix C and its decomposition B, D
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Main loop
        while calls < total_budget and not check_time():
            # Generate population
            pop = np.empty((N, dim))
            f_vals = np.empty(N)

            # Generate all N points
            for i in range(N):
                if calls >= total_budget or check_time():
                    break
                z = rng.standard_normal(dim)
                # Transform z using B and D
                x_i = x_mean + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i

                f_i = func(x_i)
                calls += 1
                f_vals[i] = f_i

                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # If we didn't complete the generation, break to avoid partial updates
            if calls >= total_budget or check_time():
                break

            # Sort population by fitness
            order = np.argsort(f_vals)
            # Select the best mu
            selected_indices = order[:mu]
            selected_pop = pop[selected_indices]

            # Weighted mean
            new_x_mean = np.sum(weights[:, None] * selected_pop, axis=0)

            # Update covariance matrix adaptation
            y = (new_x_mean - x_mean) / sigma
            pc = (1 - c_c) * pc + c_c * y
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (new_x_mean - x_mean)

            # Check for sigma adaptation (hsig)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1)))) < (1.4 + 2 / (N + 1)) * chi_N

            if hsig:
                pc = (1 - c_c) * pc + c_c * y
                delta_heav = 0
            else:
                delta_heav = c_c * (2 - c_c)

            # Update C
            # Calculate rank-mu update
            diff = selected_pop - x_mean
            rank_mu_update = np.einsum('i,ik,il->kl', weights, diff, diff)

            C = (1 - c_1 - c_mu + delta_heav) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * delta_heav * C) + c_mu * rank_mu_update

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2.0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))

            # Update mean
            x_mean = new_x_mean

            # Recompute B and D from C
            # Eigendecomposition
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))

            # Check for stagnation: if sigma is very small, restart with larger sigma
            if sigma < 1e-4:
                # Try a few random restarts
                for _ in range(5):
                    if calls >= total_budget or check_time():
                        break
                    x_restart = rng.uniform(-4, 4, size=dim)
                    x_restart = np.clip(x_restart, -5, 5)
                    fx = func(x_restart)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_restart.copy()
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
