"""Combined optimizer: LHS initialization + CMA-ES-like with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube-style sampling to find a good starting point
        # Use a small number of LHS points to explore the space
        n_lhs = min(20, budget // 10)
        lhs_points = []
        for i in range(n_lhs):
            # Generate a point in [-5, 5]^dim using stratified sampling
            point = np.array([(i + rng.random()) / n_lhs * 10 - 5 for _ in range(dim)])
            lhs_points.append(point)

        # Evaluate all LHS points
        best_x = None
        best_f = np.inf
        for p in lhs_points:
            fp = func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Check deadline
        if time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: CMA-ES-like optimization starting from best LHS point
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Initial step size

        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.log((dim - 1) / 2))) / (dim + 4)
        c_c = (4 + dim / 4) / (dim + 4 + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((dim / 2 + 1) - 2) / ((dim + 2) ** 2 + dim))
        mu = dim // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = 1.0
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = n_lhs
        for iteration in range(budget - n_lhs):
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Generate lambda = dim + 2 offspring
            lam = dim + 2
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring.append((y, i))

            # Evaluate offspring
            for y, i in offspring:
                calls_made += 1
                fy = func(y)
                offspring[i] = (y, fy)

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])

            # Update mean using weighted recombination
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += w[i] * offspring[i][0]
            x = new_x
            fx = offspring[0][1]

            # Update covariance matrix adaptation
            y_vec = (x - new_x) / sigma if sigma > 0 else np.zeros(dim)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_vec
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_vec

            delta_d = (h_sigma - 1) * c_1 * (1 + c_mu)
            B, D = B, D  # Simplified: keep diagonal for speed
            # In a full CMA-ES, we'd update B and D via eigendecomposition
            # For speed, we use a simplified adaptation

            # Update sigma
            sigma *= np.exp(c_sigma / d_amps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
