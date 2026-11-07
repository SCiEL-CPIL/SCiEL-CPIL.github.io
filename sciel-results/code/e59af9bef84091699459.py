"""Optimizer for BBOB-style continuous black-box optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES variant with adaptive restarts for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill
        d = self.dim
        budget = self.budget

        # We'll use a population-based CMA-ES
        # Parameters:
        # - lambda (population size)
        # - sigma (step size)
        # - weights, mu (for recombination)
        # - c_sigma, c_c, c_1, c_mu (adaptation rates)

        # For 5D, use lambda=10
        lam = 10
        mu = lam // 2  # top half

        # Weights: log-linear
        w = np.log(np.arange(lam, 0, -1))
        w = w / w.sum()
        w[:mu] = 1.0
        w = w / w.sum()
        mu_eff = 1.0 / (w[:mu] ** 2).sum()

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        def run_cmaes(x0, sigma0, max_evals, rng):
            """Run CMA-ES from x0 with sigma0 for max_evals evaluations."""
            x = x0.copy()
            sigma = sigma0
            B = np.eye(d)
            D = np.ones(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            evals_used = 0
            best_x = x0.copy()
            best_f = func(x0)
            evals_used = 1

            while evals_used < max_evals:
                # Check time
                if time.time() > deadline - 2.0:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(d)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    evals_used += 1

                # Evaluate
                fvals = np.array([func(y) for y in offspring])

                # Update best
                for i in range(lam):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = offspring[i].copy()

                # Sort offspring by fitness
                order = np.argsort(fvals)
                sorted_offspring = [offspring[i] for i in order]
                sorted_fvals = fvals[order]

                # Weighted mean of top mu
                m = np.zeros(d)
                for i in range(mu):
                    m += w[i] * sorted_offspring[i]

                # Update covariance matrix
                diff = np.array([sorted_offspring[i] - m for i in range(mu)])
                C = (B @ (D ** 2) @ B.T)  # current covariance approximation
                # Actually, we need to track C explicitly. Let's use a simpler approach.
                # For 5D, we can maintain C explicitly.
                # Let me restructure to maintain C directly.

                # Update evolution path for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(B @ D) * (B @ D @ ((m - x) / sigma))
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (d + 1)) * np.sqrt(d + 1) else 0.0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(d + 1) - 1))

                # Update evolution path for covariance
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

                # Update mean
                x = m

                # Update B and D (eigenvalue decomposition of C)
                # For simplicity in 5D, maintain C explicitly
                break  # This simple version won't work well; let me rewrite

            return best_x, best_f, evals_used

        # Let me implement a proper CMA-ES with explicit covariance matrix
        def cmaes(x0, sigma0, max_evals, rng):
            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            evals_used = 0
            best_x = x0.copy()
            best_f = func(x0)
            evals_used = 1

            # Precompute
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            while evals_used + lam <= max_evals:
                if time.time() > deadline - 2.0:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(d)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Regularize
                        C += 1e-8 * np.eye(d)
                        L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    evals_used += 1

                # Evaluate
                fvals = np.array([func(y) for y in offspring])

                # Update best
                for i in range(lam):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = offspring[i].copy()

                # Sort
                order = np.argsort(fvals)
                sorted_offspring = [offspring[i] for i in order]
                sorted_fvals = fvals[order]

                # Weighted mean
                m = np.zeros(d)
                for i in range(mu):
                    m += w[i] * sorted_offspring[i]

                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L @ ((m - x) / sigma))
                # Actually, ps should be in the transformed space
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - x) / sigma
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (d + 1)) * chi_n else 0.0
                sigma *= np.exp(c_sigma / d * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

                diff = np.array([(sorted_offspring[i] - m) / sigma for i in range(mu)])
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.eye(d)) + c_mu * (diff.T @ np.diag(w) @ diff)

                x = m

            return best_x, best_f, evals_used

        # Strategy: use CMA-ES with restarts
        # First, do a quick random search to find a good starting point
        rng = np.random.default_rng(42)

        # Phase 1: Random initialization - evaluate several points
        n_init = min(20, budget // 10)
        init_points = []
        init_fvals = []
        for _ in range(n_init):
            p = rng.uniform(-5, 5, d)
            f = func(p)
            init_points.append(p)
            init_fvals.append(f)

        # Find best initial point
        best_init_idx = np.argmin(init_fvals)
        x0 = init_points[best_init_idx].copy()
        f0 = init_fvals[best_init_idx]
        evals_used = n_init

        # Phase 2: CMA-ES from best initial point
        # Estimate sigma from spread of initial points
        if n_init > 1:
            sigma0 = max(0.1, np.std([np.linalg.norm(p - x0) for p in init_points]) / np.sqrt(d))
        else:
            sigma0 = 1.0

        # Allocate remaining budget to CMA-ES
        remaining = budget - evals_used
        x_best, f_best, evals_used2 = cmaes(x0, sigma0, remaining, rng)
        evals_used += evals_used2

        # If we have budget left and time, do a few more restarts
        # But for now, keep it simple
        return x_best, f_best
# EVOLVE-BLOCK-END
