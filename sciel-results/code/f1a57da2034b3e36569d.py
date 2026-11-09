"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Portfolio CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Runs multiple independent CMA-ES restarts with diverse starting points.
    Each restart adapts step size and covariance matrix to the landscape.
    Wall-clock deadline guard ensures we return before the 330s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def safe_call(x):
            nonlocal best_x, best_f
            try:
                fx = func(x)
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # CMA-ES implementation
        def cma_es(x0, fx0, n_calls, seed):
            nonlocal best_x, best_f
            rng = np.random.default_rng(seed)
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Weights: log-linear
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = int(mu * 0.8)  # effective mu for covariance update
            # Step size
            sigma = 0.5 * (max(0, np.linalg.norm(x0)) + 1.0)
            sigma = np.clip(sigma, 0.1, 2.0)
            # Covariance
            C = np.eye(n)
            Cinv_sqrt = np.eye(n)
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Rates
            c_sigma = (mueff + 2) / ((n + 1.3) * (n + 1.3) + mueff)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_covpos = (2 + 2 * mueff / n) / (n + 1.5 + 2 * mueff / n)
            c_covneg = (5 - mueff / n) / (n + 1.5 + 5 * mueff / n)
            # Chi-squared correction
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

            calls_used = 0
            x_best_local = x0.copy()
            fx_best_local = fx0

            for gen in range(2000):
                if calls_used >= n_calls:
                    break
                if time.time() >= deadline:
                    break

                # Generate population
                B = np.linalg.cholesky(C)
                offsets = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    offsets.append(x0 + sigma * (Cinv_sqrt @ z))
                offsets = np.array(offsets)
                # Clip to box
                offsets = np.clip(offsets, -5, 5)

                # Evaluate
                fvals = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_used >= n_calls:
                        break
                    if time.time() >= deadline:
                        break
                    fvals[i] = safe_call(offsets[i])
                    calls_used += 1

                # Sort
                order = np.argsort(fvals)
                fvals_s = fvals[order]
                offsets_s = offsets[order]

                # Update best
                if fvals_s[0] < fx_best_local:
                    fx_best_local = fvals_s[0]
                    x_best_local = offsets_s[0].copy()

                # Mean update
                mean_old = x0.copy()
                x0 = np.sum(w[:, None] * offsets_s[:mu], axis=0)

                # Step size control (1/5 rule via evolution path)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * \
                     (Cinv_sqrt @ ((x0 - mean_old) / sigma))
                norm_ps = np.linalg.norm(ps)
                h_sig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1)))) \
                        < (1.4 + 2 / (n + 1)) * chi_n
                if h_sig:
                    pc = (1 - c_c) * pc + c_c * (x0 - mean_old) / sigma
                else:
                    pc = (1 - c_c) * pc

                # Covariance update
                delta_h = (1 - h_sig) * (1 - (1 - c_c) ** 2)
                rank_one = np.outer(pc, pc)
                # Rank-mu
                y = (offsets_s[:mueff] - mean_old) / sigma
                rank_mu = np.sum(w[:mueff, None, None] * np.outer(y, y), axis=0)
                C = (1 + c_sigma * c_covpos * delta_h * (c_c * (2 - c_c)) + c_sigma * c_covneg - c_sigma) * C \
                    + c_sigma * c_covpos * delta_h * (c_c * (2 - c_c)) * rank_one \
                    + c_sigma * c_covneg * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Step size update
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update Cinv_sqrt periodically
                if calls_used % (2 * lam) < lam:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-12, 1e12)
                        Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    except np.linalg.LinAlgError:
                        pass

            return x_best_local, fx_best_local, calls_used

        # Portfolio: multiple restarts with diverse starts
        n_restarts = 5
        # Allocate budget: give more to later restarts (they start better)
        # Total budget for function calls
        # Reserve 1 call for initial evaluation
        remaining_budget = budget - 1

        # Initial random sample
        rng_init = np.random.default_rng(42)
        x_init = rng_init.uniform(-4, 4, dim)
        fx_init = safe_call(x_init)
        calls_so_far = 1

        # Starting points: best found + random perturbations
        start_points = [x_init.copy()]
        for i in range(1, n_restarts):
            # Mix of random and perturbed best
            if i % 2 == 0:
                sp = x_init + rng_init.standard_normal(dim) * 0.5
            else:
                sp = rng_init.uniform(-4.5, 4.5, dim)
            sp = np.clip(sp, -5, 5)
            start_points.append(sp)

        # Allocate sub-budgets: later restarts get more
        weights = np.linspace(0.5, 1.5, n_restarts)
        weights /= weights.sum()
        sub_budgets = (remaining_budget * weights).astype(int)
        # Ensure at least 10 calls per restart
        sub_budgets = np.maximum(sub_budgets, 10)
        # Adjust to not exceed remaining
        total_alloc = sub_budgets.sum()
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            sub_budgets = (sub_budgets * scale).astype(int)
            sub_budgets = np.maximum(sub_budgets, 5)

        for i in range(n_restarts):
            if calls_so_far >= budget:
                break
            if time.time() >= deadline - 5.0:
                break

            sp = start_points[i]
            # Evaluate starting point if not already done
            if i == 0:
                fx_sp = fx_init
                calls_sp = 0
            else:
                fx_sp = safe_call(sp)
                calls_sp = 1
                calls_so_far += 1

            # Run CMA-ES
            x_best_c, fx_best_c, calls_used = cma_es(sp, fx_sp, sub_budgets[i] - calls_sp, seed=100 + i)
            calls_so_far += calls_used

            if fx_best_c < best_f:
                best_f = fx_best_c
                best_x = x_best_c.copy()

        # Final: if we have budget left, do a local polish
        if calls_so_far < budget and time.time() < deadline - 2.0:
            # Simple local refinement around best
            x_polish = best_x.copy()
            fx_polish = best_f
            sigma_p = 0.05
            for _ in range(min(50, budget - calls_so_far)):
                if time.time() >= deadline - 1.0:
                    break
                y = np.clip(x_polish + sigma_p * rng_init.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                calls_so_far += 1
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    sigma_p *= 1.2
                else:
                    sigma_p *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
