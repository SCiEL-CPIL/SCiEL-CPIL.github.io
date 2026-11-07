"""Optimizer for BBOB f14 (Different powers) on [-5,5]^5 using CMA-ES + polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time deadline guard and final local polish pass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0

        rng = np.random.default_rng(42)

        # --- CMA-ES setup ---
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2.0 - 1.0 / (dim + 1.3) + 2.0 * min(1, (mueff - 2) / (dim + 2))) / (
            (dim + 1.3) ** 2 + 2.0
        )
        cmu = min(
            1.0 - c1,
            2.0 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + 2.0),
        )
        damps = 1.0 + 2.0 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Initial state: center at origin, sigma based on box
        xmean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)  # Cholesky of C (initially I)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Evaluate initial center
        xmean_clipped = np.clip(xmean, lb, ub)
        fvals = np.full(lam, np.inf)
        fvals[0] = func(xmean_clipped)
        best_x = xmean_clipped.copy()
        best_f = fvals[0]
        calls_made = 1

        # Precompute weights for update
        # We'll use the standard CMA-ES update

        def update_chol(C, C_sqrt):
            """Recompute Cholesky of C, ensuring positive definite."""
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Add small regularization
                C_reg = C + 1e-8 * np.eye(dim)
                L = np.linalg.cholesky(C_reg)
                return L

        for gen in range(budget // lam + 2):
            if calls_made >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Sample population
            ys = []
            for i in range(lam):
                if i == 0:
                    y = xmean  # reuse center evaluation
                else:
                    z = rng.standard_normal(dim)
                    y = xmean + sigma * C_sqrt @ z
                    y = np.clip(y, lb, ub)
                    fy = func(y)
                    calls_made += 1
                    ys.append((y, fy))
            # Build full population list
            pop = [(xmean_clipped, fvals[0])] + ys

            # Sort by fitness
            pop.sort(key=lambda p: p[1])

            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Weighted mean of top mu
            new_xmean = np.zeros(dim)
            for i in range(mu):
                new_xmean += w[i] * pop[i][0]

            # Standard deviation
            yvec = (new_xmean - xmean) / sigma
            ys_norm = np.linalg.norm(yvec)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * yvec
            hsig = (
                np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                < (1 + 2.0 / (dim + 1)) * np.sqrt(dim + 1) / np.sqrt(dim - 1)
            )
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * yvec
                delta_hs = (
                    1 - cc
                ) / (1 - (1 - cc) ** (gen + 1))
            else:
                delta_hs = 0.0

            # Update covariance
            delta = np.outer(pc, pc) - (
                1 - (1 - cc) ** (gen + 1)
            ) * C
            C = (
                (1 + c1 * delta_hs - c1 - cmu) * C
                + c1 * delta
                + cmu / sigma ** 2
            )
            # Build rank-mu update
            zmat = np.array([(pop[i][0] - xmean) / sigma for i in range(mu)])
            rank_mu = np.outer(zmat.T @ w, zmat)
            C += cmu * rank_mu / sigma ** 2

            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update sigma
            sigma *= np.exp(
                cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1)
            )
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update mean
            xmean = new_xmean
            xmean = np.clip(xmean, lb, ub)

            # Update Cholesky
            C_sqrt = update_chol(C, C_sqrt)

        # --- Final polish pass: local refinement with small steps ---
        if calls_made < budget and time.time() < deadline - 2.0:
            x_polish = best_x.copy()
            f_polish = best_f
            # Use a very small step size for local refinement
            sigma_polish = max(sigma * 0.01, 1e-6)
            remaining = budget - calls_made
            # Use a simple (1+1) with tiny steps for final polish
            for _ in range(remaining):
                if time.time() > deadline - 1.0:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x_polish + sigma_polish * z, lb, ub)
                fy = func(y)
                calls_made += 1
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.9
                    sigma_polish = max(sigma_polish, 1e-8)
            if f_polish < best_f:
                best_x = x_polish
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
