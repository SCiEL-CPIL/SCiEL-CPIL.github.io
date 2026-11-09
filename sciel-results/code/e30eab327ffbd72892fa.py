"""Rewritten optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Global best tracking
        best_x = np.full(n, np.nan)
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def run_cmaes(x0, sigma0, max_calls):
            nonlocal call_count, best_x, best_f
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evals = 0
            last_best_f = np.inf
            no_improve = 0

            for gen in range(max_calls // lambda_ + 1):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                # Use eigen decomposition of C for sampling
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eigvals = np.sqrt(eigvals)
                B = eigvecs
                D = sqrt_eigvals

                pop = np.zeros((lambda_, n))
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    xi = x + sigma * (B @ (D * z))
                    fvals[i] = eval_point(xi)
                    pop[i] = xi
                    evals += 1

                # Sort by fitness
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]

                # Update mean
                x_new = np.sum(w[:, None] * pop[:mu], axis=0)

                # Update step-size control (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma * np.linalg.norm(x_new - x)) * (B @ (D * (x_new - x)))
                # Actually, standard CMA-ES: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff)/sigma * (B*D^-1 * (x_new - x))
                # Let me redo this properly
                # x_new - x is in the transformed space
                diff = x_new - x
                # Transform to normal space: D^-1 B^T (x_new - x)
                try:
                    D_inv = 1.0 / np.maximum(D, 1e-12)
                    normal_diff = D_inv * (B.T @ diff)
                except:
                    normal_diff = diff / sigma

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * normal_diff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals + 1) / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 1e8)

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * normal_diff

                # Update covariance
                delta = hsig * (2 - cc)
                C = (1 + c1 * delta - c1 - cmu) * C
                for i in range(mu):
                    zi = D_inv * (B.T @ (pop[i] - x))
                    C += cmu * w[i] * np.outer(zi, zi)
                C += c1 * (np.outer(pc, pc) + delta - cc * (2 - cc)) * np.eye(n)

                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    eigvals_c, eigvecs_c = np.linalg.eigh(C)
                    eigvals_c = np.maximum(eigvals_c, 1e-12)
                    C = (eigvecs_c * eigvals_c) @ eigvecs_c.T
                    C = (C + C.T) / 2
                except:
                    C = np.eye(n)

                x = x_new

                # Check improvement
                if fvals[0] < last_best_f - 1e-10:
                    last_best_f = fvals[0]
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 20:
                    break

                if evals >= max_calls:
                    break

        # Restart strategy: multiple restarts with different initializations
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break

            if r == 0:
                x0 = np.zeros(n)
                sigma0 = 1.0
            else:
                x0 = rng.uniform(-3, 3, n)
                sigma0 = 1.5

            # Run CMA-ES with remaining budget
            remaining = budget - call_count
            max_calls = min(calls_per_restart, remaining)
            run_cmaes(x0, sigma0, max_calls)

            # Fine-tuning phase with smaller sigma
            if call_count < budget and time.time() < deadline:
                if not np.isnan(best_f) and np.isfinite(best_f):
                    x0 = best_x.copy()
                    sigma0 = max(0.1, 1.0 / (1 + call_count / 50))
                    remaining = budget - call_count
                    max_calls = min(remaining, 100)
                    run_cmaes(x0, sigma0, max_calls)

        # Final local search with small steps if time remains
        if call_count < budget and time.time() < deadline and not np.isnan(best_f):
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, budget - call_count)):
                if call_count >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = eval_point(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
