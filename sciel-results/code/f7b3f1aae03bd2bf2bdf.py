"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and adaptive step size for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(7)

        best_x = None
        best_f = float('inf')

        # Strategy: CMA-ES with restarts
        # Use a smaller population for CMA-ES to fit within budget
        # Budget allocation: 3 restarts with decreasing budgets
        # CMA-ES typically needs ~300*d evaluations to converge
        # With dim=5, that's ~1500 evals, so we need multiple restarts with smaller pops

        # Use (lambda, mu) = (9, 3) CMA-ES variant
        lambda_pop = 9  # population size
        mu = lambda_pop // 2  # number of recombination parents

        # Budget allocation for restarts
        # First restart gets more budget for exploration
        # Later restarts focus on exploitation
        n_restarts = 3
        budgets = [200, 180, 120]
        # Adjust to fit budget
        total_alloc = sum(budgets)
        scale = budget / total_alloc
        budgets = [max(lambda_pop + 1, int(b * scale)) for b in budgets]
        # Ensure we don't exceed budget
        while sum(budgets) > budget:
            budgets[-1] -= 1
        while sum(budgets) < budget:
            budgets[0] += 1

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if budget - sum(budgets[:restart]) <= 0:
                break

            # Diverse initial points
            if restart == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            elif restart == 1:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.5
            else:
                # Perturb around best found so far
                if best_x is not None:
                    x0 = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                    sigma0 = 0.8
                else:
                    x0 = rng.uniform(-3, 3, dim)
                    sigma0 = 1.5

            fx0 = func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # CMA-ES implementation
            # Population mean
            mean = x0.copy()
            sigma = sigma0
            b = budgets[restart]

            # Initialize CMA-ES matrices
            C = np.eye(dim)  # covariance matrix
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))

            # CMA-ES parameters
            # Learning rates
            cs = (mu + 2) / (dim + mu + 5)
            cmu = 2 / ((dim + 2) ** 2 + mu)
            cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu)
            cmu = cmu - c1

            # Damping
            # chiN = expected norm of N(0,I)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Number of evaluations for this restart
            evals_used = 1  # initial evaluation
            n_iter = b - 1

            # Sort indices for recombination
            # We'll use weighted recombination

            for it in range(n_iter):
                if time.time() > deadline:
                    break
                if evals_used >= b:
                    break

                # Generate population
                pop = np.zeros((lambda_pop, dim))
                fit = np.zeros(lambda_pop)

                for i in range(lambda_pop):
                    # Sample from N(mean, sigma^2 * C)
                    # Use Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset to identity
                        C = np.eye(dim)
                        L = np.eye(dim)

                    # Standard normal sample
                    z = rng.standard_normal(dim)
                    # Transform to N(0, C)
                    y = L @ z
                    # Clip to bounds
                    xi = np.clip(mean + sigma * y, -5, 5)
                    pop[i] = xi
                    fit[i] = func(xi)
                    evals_used += 1

                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = xi.copy()

                # Sort by fitness
                order = np.argsort(fit)
                fit_sorted = fit[order]
                pop_sorted = pop[order]

                # Update mean
                # Weighted recombination
                weights = np.zeros(mu)
                # Logarithmic weights
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = w / w.sum()

                mean_old = mean.copy()
                mean = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)

                # Update covariance matrix
                diff = mean - mean_old
                # Rank-one update
                C += c1 * (diff[:, None] * diff[None, :] - C)

                # Rank-mu update
                diff_mu = pop_sorted[:mu] - mean_old
                # Weighted outer products
                rank_mu = np.zeros((dim, dim))
                for k in range(mu):
                    rank_mu += weights[k] * np.outer(diff_mu[k], diff_mu[k])
                C += cmu * rank_mu

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Update evolution paths
                # ps update
                inv_sigma = 1.0 / sigma
                ps = (1 - cs) * ps + cs * inv_sigma * (L.T @ (mean - mean_old) / sigma)

                # pc update
                pc = (1 - cc) * pc + cc * cs / (1 - cs) * (
                    (1 + cs) * (mean - mean_old) / sigma - ps
                )

                # sigma update
                # Hessian correction
                hsig = (
                    np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))
                )
                hsig = 1.0 if hsig < (1.4 + 2 / (dim + 1)) * chiN else 0.0

                # Update C with pc
                C += (c1 * (hsig - 1 + cc) + cmu * (hsig - 1)) * (
                    cc / (1 - cc) * C + pc
                )

                # Sigma update
                sigma *= np.exp(
                    cs / chiN * (np.linalg.norm(ps) / chiN - 1)
                )

                # Keep sigma within reasonable bounds
                if sigma < 1e-6:
                    sigma = 1e-6
                if sigma > 15.0:
                    sigma = 15.0

                # Periodically re-orthogonalize C to prevent numerical issues
                if it % 50 == 0:
                    # Eigenvalue decomposition and clamp
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-8)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
