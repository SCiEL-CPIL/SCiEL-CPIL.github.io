"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f11 (Discus) with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Multi-start CMA-ES
        # For dim=5, use a population size that balances exploration/exploitation
        # Standard CMA-ES: lambda = 4 + floor(3 * log(dim))
        lam = max(4, 4 + int(3 * np.log(dim)))
        # For dim=5: lam = 4 + floor(3*1.609) = 4 + 4 = 8

        # We'll split the budget among multiple restarts
        # Early restarts help escape local minima in the discus landscape
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            # Check time
            if time.time() > deadline - 10:
                break

            # Initialize center
            if restart == 0:
                # Start near origin (likely to be near the shifted minimum)
                c = np.zeros(dim)
            else:
                # Use best found so far with some perturbation
                if best_x is not None:
                    c = best_x + np.random.default_rng(restart).uniform(-1, 1, dim)
                    c = np.clip(c, -5, 5)
                else:
                    c = np.random.default_rng(restart).uniform(-3, 3, dim)

            # CMA-ES parameters
            sigma = 2.0  # Initial step size
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Step sizes for paths
            cs = (1 + 2 * max(0, np.sqrt((lam / 2 - 1)) / (dim + 2) - 1)) / (dim + 2 + 2)
            cmu = (2 + (lam / 2 - 2) / ((dim + 2) ** 2 + 1)) / (dim + 4)
            # Damping for pc
            cs_pc = (1 + 2 * max(0, np.sqrt((lam / 2 - 1)) / (dim + 2) - 1)) * (2 * cmu)
            # Chi-lambda for normal distribution
            chi_lam = lam ** 0.5 * (1 - 1 / (4 * lam) - 1 / (24 * lam ** 2))

            # Generate initial population
            rng = np.random.default_rng(42 + restart)

            # We'll use a simple approach: generate lambda points, evaluate, select top half
            # For efficiency with limited budget, we'll do fewer generations with larger steps

            calls_in_restart = 0
            max_calls = budget_per_restart

            # Initial evaluation
            c_eval = c.copy()
            fc = func(c_eval)
            calls_in_restart += 1
            if fc < best_f:
                best_f = fc
                best_x = c_eval.copy()

            # Track history for covariance
            # We'll use a simplified CMA-ES that updates the distribution each generation

            for gen in range(max_calls // lam + 1):
                if time.time() > deadline - 5:
                    break
                if calls_in_restart >= max_calls:
                    break

                # Generate population
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes non-positive definite
                    C = np.eye(dim)
                    L = np.eye(dim)

                # Generate lambda points
                W = rng.standard_normal((lam, dim))
                W = np.linalg.solve(L.T, W.T).T  # Transform: W @ L^{-T}
                # Actually we want z ~ N(0, C), so x = c + sigma * (L @ z)
                # Let's redo: if C = L @ L.T, then z @ L gives covariance C
                # So x_i = c + sigma * (z_i @ L) where z_i ~ N(0, I)

                X = c + sigma * (W @ L)
                X = np.clip(X, -5, 5)

                # Evaluate population
                F = np.array([func(x) for x in X])
                calls_in_restart += lam

                # Sort by fitness
                order = np.argsort(F)
                X = X[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()

                # Selection weights (interpolated rank weights)
                mu = lam // 2
                # Weights for the top mu individuals
                w = np.zeros(mu)
                # Use simple linear weights
                w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mueff = 1 / np.sum(w ** 2)

                # Update center
                c_new = np.sum(w[:, None] * X[:mu], axis=0)

                # Update evolution path ps
                ps = (1 - cs) * ps + cs / chi_lam * np.sum(w[:, None] * (X[:mu] - c) / sigma, axis=0)

                # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^2) < (1.4 + 2/(dim+1)) * chi_lam
                hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** gen) < (1.4 + 2 / (dim + 1)) * chi_lam else 0.0

                # Update evolution path pc
                pc = (1 - cs_pc) * pc + hsig * cs_pc * (c_new - c) / sigma

                # Update covariance matrix
                # Delta h for rank-one update
                delta_h = (1 - hsig) * (1 - (1 - cs_pc) ** (gen + 1))
                delta_h = 2 * delta_h - 1 if delta_h > 0 else delta_h

                # Rank-one update
                C = (1 + cmu * mueff * delta_h - cmu) * C
                C += cmu * np.outer(c_new - c, c_new - c) / sigma ** 2

                # Rank-mu update
                Y = (X[:mu] - c) / sigma
                C -= cmu * np.sum(w[:, None] * (Y[:, :, None] * Y[:, None, :]), axis=0)

                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                # Add small regularization
                C += 1e-12 * np.eye(dim)

                # Update sigma
                # Use the ratio of norms
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp(cs / (dim + 2) * (ps_norm / chi_lam - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)

                # Update center
                c = c_new

            # End of this restart

        # Final local refinement around best point
        if time.time() < deadline - 5 and best_x is not None:
            # Do a few more evaluations around the best point with small steps
            rng_final = np.random.default_rng(123)
            local_sigma = 0.1
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                x_trial = np.clip(best_x + local_sigma * rng_final.standard_normal(dim), -5, 5)
                f_trial = func(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                    local_sigma *= 1.5
                else:
                    local_sigma *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
