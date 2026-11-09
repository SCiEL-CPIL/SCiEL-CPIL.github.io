"""Rewritten optimizer: multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


class Optimizer:
    """Multi-start CMA-ES for Rastrigin-rotated f15 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        best_x = np.full(dim, np.nan)
        best_f = np.inf

        # CMA-ES implementation with vectorization and restarts
        def cmaes(func, x0, sigma0, budget_cma, dim, deadline):
            n = dim
            # Population size
            m = max(4, int(round((4 + int(np.log(n))) * n)))
            # CMA-ES parameters
            cmac = 1.0 / (n + 4)
            cmam = 1.0 / (n + 4)
            c1 = (2 / ((n + 1.3) ** 2 + m)) * (1 - cmac)
            cmu = min(0.75 - c1, 2 * (2 - (n + 2) / (n + 4)) / ((n + 2) + m))
            cs = (m + 1.7) / (n + m + 5)
            ds = 1 + 2 * max(0, np.sqrt((m - 1) / (n + 1)) - 1) + cs
            cc = (4 + n / m) / (4 + n + m / n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Initialize
            x = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)

            # Initial function eval
            fx = func(np.clip(x, -5, 5))
            best_local_x = x.copy()
            best_local_f = fx

            count = 1
            mu = m // 2
            w = np.zeros(m)
            w[:mu] = np.log(m / 2) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w[:mu] ** 2)

            no_improve = 0
            stagnation_limit = 10  # Restart if no improvement for 10 gens

            for gen in range(budget_cma):
                if time.time() > deadline:
                    break

                # Vectorized sampling
                # z: (m, n)
                Z = np.random.standard_normal((m, n))
                # Y = x + sigma * (B * (D * Z))
                # B @ (D * Z.T).T
                # Z is (m, n). D is (n,). B is (n, n).
                # We want Y[i] = x + sigma * B @ (D * Z[i])
                # Let Z_mat = Z.T (n, m).
                # D * Z_mat -> (n, m)
                # B @ (D * Z_mat) -> (n, m)
                # Transpose back to (m, n)
                Y = x + sigma * (B @ (D * Z.T)).T
                Y = np.clip(Y, -5, 5)

                # Evaluate
                fvals = np.array([func(y) for y in Y])
                
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_local_f:
                    best_local_f = fvals[min_idx]
                    best_local_x = Y[min_idx].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                count += m

                # Check for stagnation and restart
                if no_improve >= stagnation_limit:
                    # Restart with larger sigma around best incumbent
                    # or random restart? Directive says "reinitializing the search distribution with a larger sigma centered on the best incumbent"
                    x = best_local_x.copy()
                    sigma = max(sigma * 2.0, 1.0)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)
                    no_improve = 0
                    continue

                # Sort by fitness
                order = np.argsort(fvals)
                Y_sorted = Y[order]
                fvals_sorted = fvals[order]

                # Update mean
                x_old = x.copy()
                # Weighted average of top mu
                x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)

                # Update step size control
                # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (B @ (D^-1)) @ ((x_new - x_old) / sigma)
                # Note: B @ (D^-1) is the inverse of the transformation used in sampling?
                # Sampling was Y = x + sigma * B @ (D * Z).
                # So Z = D^-1 @ B^-1 @ (Y - x) / sigma.
                # B is orthogonal (from eigh), so B^-1 = B.T.
                # (x_new - x_old) is in the "Y space".
                # We need to transform (x_new - x_old) back to Z space for ps update?
                # Standard CMA-ES: ps is in the "step size" space.
                # The update formula is:
                # ps = (1-cs)ps + sqrt(cs(2-cs)mu_eff) * B^{-1} D^{-1} (x_new - x_old)/sigma
                # Wait, B and D define the covariance.
                # Let's stick to the standard implementation which often avoids explicit B, D inversion if possible,
                # but here we have them.
                # B @ (D * Z) = Y_offset.
                # So Y_offset = sigma * B @ (D * Z).
                # Z = D^-1 @ B.T @ (Y_offset / sigma).
                # x_new - x_old is the mean shift.
                # The "natural" gradient direction in Z-space is D^-1 @ B.T @ (x_new - x_old) / sigma.
                
                step = (x_new - x_old) / sigma
                # Transform to Z-space
                step_z = np.linalg.solve(D, B.T @ step) # D^-1 B.T step
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * step_z
                
                pc = (1 - cmac) * pc + cmac * ((1 - cmam) * pc + np.sqrt(cmac * (2 - cmac) * mu_eff) * step)

                # Update covariance
                delta_sig = np.linalg.norm(ps) / chi_n - 1
                hsig = 1.0 if (delta_sig ** 2) / (1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (n + 1)) * (1 - (1 - cs) ** (2 * (gen + 1))) else 0.0
                rank1 = np.outer(pc, pc)
                
                # rankmu: sum w_i * z_i * z_i^T
                # Z_sorted = Z[order]
                Z_sorted = Z[order]
                rankmu = np.sum(w[:mu, np.newaxis, np.newaxis] * Z_sorted[:mu, :, np.newaxis] * Z_sorted[:mu, np.newaxis, :], axis=0)
                
                C = (1 + c1 * hsig + cmu * hsig - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                # Update B and D
                C_ = 0.5 * (C + C.T)  # symmetrize
                D, B = np.linalg.eigh(C_)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B[:, ::-1]
                D = D[::-1]

                x = x_new

                if count >= budget_cma:
                    break

            return best_local_x, best_local_f, count

        # Multi-start strategy
        np.random.seed(42)

        total_calls = 0
        # We will use a loop of CMA-ES runs.
        # Each run gets a portion of the budget.
        # If we run out of budget, we stop.
        
        # Strategy: 
        # 1. Start at 0.
        # 2. If no significant improvement, restart at random points.
        
        n_starts = 10
        budget_per_start = budget // n_starts
        
        for s in range(n_starts):
            if time.time() > deadline:
                break
            if total_calls >= budget:
                break

            if s == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            else:
                # Random restart
                x0 = np.random.uniform(-5, 5, dim)
                sigma0 = 1.5

            remaining = budget - total_calls
            budget_this = min(budget_per_start, remaining)
            if budget_this < 10:
                break

            res_x, res_f, calls_used = cmaes(func, x0, sigma0, budget_this, dim, deadline)
            total_calls += calls_used

            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
