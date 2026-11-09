"""Rewritten optimizer: multi-start CMA-ES with time guard."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin under 300s
        rng = np.random.default_rng(42)
        d = self.dim
        total_budget = self.budget

        best_x = np.zeros(d)
        best_f = np.inf
        calls_used = 0

        def _clip(x):
            return np.clip(x, -5.0, 5.0)

        def _eval(x):
            nonlocal calls_used
            x = _clip(np.asarray(x, dtype=float))
            f = float(func(x))
            calls_used += 1
            return x, f

        # Strategy: run several CMA-ES restarts with different initializations.
        # Each restart gets a fraction of the remaining budget.
        # CMA-ES parameters: population size lambda, step size sigma.

        def cma_es(x0, sigma0, budget_for_run, rng_local):
            """Run CMA-ES for up to budget_for_run evaluations."""
            nonlocal best_x, best_f, calls_used
            nonlocal deadline

            d = self.dim
            lam = max(4, int(4 + 2 * np.log(d)))  # population size
            x_mean = np.array(x0, dtype=float).copy()
            sigma = sigma0
            B = np.eye(d)
            C = np.eye(d)
            pc = np.zeros(d)
            cs = 1.0
            cc = (4 + lam / d) / (d + 4 + 2 * lam / d)
            cmu = 2 / ((d + 1.3) ** 2 + lam)
            c_sigma = (4 + lam / d) / (d + 4 + 3 * lam / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (d + 1)) - 1) + c_sigma

            # Precompute
            for i in range(d):
                pass  # C initialized as identity

            # We'll use a simplified CMA-ES update
            w = np.log(lam / 2) - np.arange(1, lam + 1)
            w = np.exp(w)
            w /= w.sum()
            mu = int(np.sum(w > 0))  # number of weights > 0
            mu_eff = 1 / np.sum(w[:mu] ** 2)

            best_local_x = x_mean.copy()
            best_local_f = np.inf

            evals_in_run = 0
            while evals_in_run < budget_for_run and calls_used < total_budget:
                if time.time() > deadline:
                    break

                # Generate population
                W = np.zeros((lam, d))
                for i in range(lam):
                    # Cholesky of C
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
                    z = rng_local.standard_normal(d)
                    W[i] = x_mean + sigma * L @ z

                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if evals_in_run + 1 > budget_for_run or calls_used >= total_budget:
                        break
                    if time.time() > deadline:
                        break
                    xi, fi = _eval(W[i])
                    F[i] = fi
                    evals_in_run += 1
                    if fi < best_local_f:
                        best_local_f = fi
                        best_local_x = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if evals_in_run >= budget_for_run or calls_used >= total_budget:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                W_sorted = W[idx]
                F_sorted = F[idx]

                # Update mean
                x_mean_new = np.sum(w[:mu, np.newaxis] * W_sorted[:mu], axis=0)
                # Update C
                y_sigma = (x_mean_new - x_mean) / sigma
                cs = (1 - c_sigma) * cs + c_sigma * (np.linalg.norm(y_sigma) /
                     np.sqrt(1 - (1 - cs) ** 2) - 1)
                pc = (1 - cc) * pc + cc * y_sigma
                # Rank-1 update
                v = x_mean_new - x_mean
                C = (1 - cmu * mu_eff) * C + cmu * np.outer(v, v) / sigma ** 2
                # Rank-mu update
                H = (W_sorted[:mu] - x_mean) / sigma
                C += cmu * (np.sum(w[:mu, np.newaxis, np.newaxis] *
                                    H[:, :, np.newaxis] * H[:, np.newaxis, :], axis=0))

                # Symmetrize
                C = 0.5 * (C + C.T)

                x_mean = x_mean_new
                # Adapt sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(y_sigma) /
                           np.sqrt(1 - (1 - cs) ** 2) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

                # Clip mean to box
                x_mean = _clip(x_mean)

            return best_local_x, best_local_f

        # Multi-start strategy
        # First few starts: center-biased; later starts: more spread
        n_starts = 5
        # Allocate budget: give more to earlier starts
        # Start 0: 40%, Start 1: 25%, Start 2: 15%, Start 3: 10%, Start 4: 10%
        fractions = [0.40, 0.25, 0.15, 0.10, 0.10]

        for i in range(n_starts):
            if calls_used >= total_budget or time.time() > deadline:
                break

            remaining = total_budget - calls_used
            budget_for_run = int(remaining * fractions[i])
            if budget_for_run < 10:
                budget_for_run = remaining

            # Initial point
            if i < 2:
                # Center-biased: closer to origin
                x0 = rng.uniform(-3, 3, d)
            elif i < 4:
                x0 = rng.uniform(-4, 4, d)
            else:
                x0 = rng.uniform(-5, 5, d)

            sigma0 = 2.0 if i < 2 else 1.5

            cma_es(x0, sigma0, budget_for_run, rng)

        # Final polish: use remaining budget for a local refinement around best_x
        if calls_used < total_budget and time.time() < deadline:
            remaining = total_budget - calls_used
            if remaining >= 5:
                # Simple local search: small random perturbations around best_x
                sigma_polish = 0.1
                for _ in range(min(remaining, 50)):
                    if calls_used >= total_budget or time.time() > deadline:
                        break
                    perturbation = rng.standard_normal(d) * sigma_polish
                    x_try = _clip(best_x + perturbation)
                    f_try = float(func(x_try))
                    calls_used += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        sigma_polish *= 1.2
                    else:
                        sigma_polish *= 0.8
                        sigma_polish = max(sigma_polish, 0.01)

        return best_x, best_f
