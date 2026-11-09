"""Two-stage optimizer for BBOB-style f19 (Griewank-Rosenbrock).

Stage 1: A compact deterministic Latin-hypercube-like coarse screen (<= 20
evaluations) to bracket the global basin quickly.
Stage 2: A single high-precision CMA-ES run initialized at the best coarse
point, exploiting the anisotropic geometry of the rotated landscape.
A deadline guard guarantees we return the best point found before the
evaluator kills the process.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse LHS screen + single CMA-ES run for f19 on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 270.0  # safety margin vs 300 s kill

        calls_made = 0
        best_x = None
        best_f = np.inf

        rng = np.random.RandomState(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
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

        # ------------------------------------------------------------------
        # Stage 1: Compact Latin-hypercube-like coarse screen.
        #   16 points spread across [-4, 4]^5 plus the origin.
        # ------------------------------------------------------------------
        def coarse_screen():
            n_coarse = 16
            # Deterministic LHS-like points
            pts = [np.zeros(dim)]
            for i in range(n_coarse):
                x = np.zeros(dim)
                for d in range(dim):
                    # Spread across [-4, 4] with a deterministic pattern
                    frac = (i * (d + 1) + d * (d + 1)) % n_coarse
                    x[d] = -4.0 + 8.0 * frac / n_coarse
                pts.append(x)
            for x in pts:
                if not time_ok():
                    break
                safe_call(x)

        # ------------------------------------------------------------------
        # Stage 2: CMA-ES
        # ------------------------------------------------------------------
        def cma_es(x0, sigma0, max_evals):
            nonlocal calls_made, best_x, best_f
            if not time_ok():
                return x0, best_f

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            evals_used = 0
            max_iter = max_evals // lam + 5

            for it in range(max_iter):
                if not time_ok() or evals_used >= max_evals:
                    break

                # Sample population
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eigvals = np.sqrt(eigvals)

                z = rng.randn(lam, n)
                y = (eigvecs * sqrt_eigvals) @ z.T
                pop = mean + sigma * y.T

                # Evaluate
                fvals = np.array([safe_call(x) for x in pop])
                evals_used += lam

                # Sort
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update covariance
                y_inv = np.linalg.inv((eigvecs * sqrt_eigvals).T)
                y_mu = (pop[:mu] - mean) / sigma
                y_mu = y_mu @ y_inv

                delta_sig = (np.sum(weights * (fvals[:mu] - fvals[mu:])) / (fvals[-1] - fvals[0] + 1e-15)) if (fvals[-1] - fvals[0] > 1e-10) else 0.0
                # Simplified delta sigma
                hsig = 1.0
                # For simplicity, use standard update
                C = (1 - (c1 + cmu)) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * np.eye(n) if False else np.outer(pc, pc))
                # Actually let's do the standard CMA-ES update properly
                # Recompute with standard formulas
                C_new = (1 - (c1 + cmu)) * C
                C_new += c1 * (2 * hsig * cc * (1 - cc) * np.eye(n) + (1 - hsig) * (np.outer(pc, pc) - cc * (1 - cc) * np.eye(n)))
                # This is getting complicated; use a simpler rank-1 + rank-mu update
                # Let's just do rank-1 + rank-mu with hsig=1
                C_new = (1 - (c1 + cmu)) * C
                C_new += c1 * (np.outer(pc, pc))
                C_new += cmu * np.sum(weights[:, None, None] * np.einsum('ij,ik->ijk', y_mu, y_mu), axis=0)
                C = C_new

                # Update ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (mean - new_mean) * 0  # placeholder
                # Properly: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * (mean - new_mean)/sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (mean - new_mean) / sigma

                # Update pc
                hsig = 1.0  # simplified
                pc = (1 - cc) * pc + hsig * cc * (mean - new_mean) / sigma

                mean = new_mean

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                # Symmetrize C
                C = (C + C.T) / 2

                # Check convergence
                if np.linalg.norm(mean - x0) < 1e-10 or sigma < 1e-15:
                    break

            return mean, fvals[0]

        # ------------------------------------------------------------------
        # Main schedule
        # ------------------------------------------------------------------
        coarse_screen()

        if time_ok() and best_x is not None:
            # Initialize CMA-ES with small sigma to exploit local structure
            sigma0 = 0.5
            remaining = budget - calls_made
            if remaining > 20:
                x, f = cma_es(best_x, sigma0, remaining - 5)
                if f < best_f:
                    best_x, best_f = x, f

        # Final polish: coordinate descent
        if time_ok() and best_x is not None:
            x = best_x.copy()
            f = best_f
            step = 0.1
            for _ in range(4):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sign in (1.0, -1.0):
                        if not time_ok():
                            break
                        cand = x.copy()
                        cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                        fc = safe_call(cand)
                        if fc < f - 1e-12:
                            x = cand
                            f = fc
                            improved = True
                            break
                if not improved:
                    break
            if f < best_f:
                best_x, best_f = x, f

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
