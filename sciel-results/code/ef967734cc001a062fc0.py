"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with proper covariance adaptation, (1+1)-ES polish,
    and time guard. Uses standard CMA-ES formulas for fast convergence on
    smooth, moderately conditioned problems."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 320.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        # Strategy: 
        # 1. Quick random initialization (10 points)
        # 2. CMA-ES from best initial point
        # 3. (1+1)-ES polish on the best found

        rng = np.random.default_rng(42)

        # Quick random initialization
        init_budget = 10
        candidates = [np.zeros(dim)]
        for _ in range(init_budget - 1):
            candidates.append(rng.uniform(-4, 4, dim))
        
        cand_vals = [func(np.clip(c, -5, 5)) for c in candidates]
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]
        used = init_budget

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # CMA-ES parameters
        lam = 12  # population size
        mu = lam // 2  # number of parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        remaining = total_budget - used
        polish_budget = 50
        cma_budget = max(0, remaining - polish_budget)
        max_iters = cma_budget // lam

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if used >= total_budget:
                break

            # Sample lambda points
            Y = np.zeros((lam, dim))
            for i in range(lam):
                # Sample from N(0, C)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                z = rng.standard_normal(dim)
                Y[i] = eigvecs @ (np.sqrt(eigvals) * z)

            # Evaluate
            X = np.clip(x[None, :] + sigma * Y, -5, 5)
            F = np.array([func(xi) for xi in X])
            used += lam

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            F_sorted = F[order]
            Y_sorted = Y[order]
            X_sorted = X[order]

            # Mean update
            x_old = x.copy()
            x = x_old + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = np.clip(x, -5, 5)

            # Covariance matrix adaptation using standard CMA-ES formula
            ys = Y_sorted[:mu]
            mean_y = np.sum(w[:, None] * ys, axis=0)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_y

            # Update C using combined Rank-1 and Rank-Mu update
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])

            coeff_c = (1 - c_c) * (1 - c_sigma)
            coeff_pc = c_c * (1 - c_c) * d_sigma
            coeff_h = c_sigma * (1 - 1.0 / mu_eff)

            C_new = coeff_c * C + coeff_pc * np.outer(p_c, p_c) + coeff_h * H
            C_new = (C_new + C_new.T) / 2

            # Ensure positive semi-definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C_new)
            if np.min(eigvals_C) < 1e-12:
                C_new = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T
            C = C_new

            # Step-size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * mean_y
            sigma *= np.exp((1.0 / (5 * dim + 5)) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 20.0)

        # (1+1)-ES polish
        if time.time() < deadline and used < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.1, sigma * 0.3)
            polish_left = total_budget - used
            for _ in range(polish_left):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
        return best_x, best_f
# EVOLVE-BLOCK-END
