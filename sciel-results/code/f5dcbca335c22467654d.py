"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restarts for BBOB-style f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Global best
        best_x = np.zeros(dim)
        best_f = float('inf')

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of selected individuals
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 - cc - cs) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cmaes_run(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with sigma0, using at most max_evals function calls."""
            nonlocal best_x, best_f

            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            evals = 0
            gen = 0

            while evals < max_evals:
                if time.time() > deadline - 5.0:
                    break
                gen += 1
                # Sample population
                B, D = np.linalg.eigh(C)
                B = B * np.sqrt(D)
                pops = [xmean + sigma * B @ rng.standard_normal(dim) for _ in range(lambda_)]
                # Evaluate
                fitvals = []
                for ind in pops:
                    x_clip = np.clip(ind, -5.0, 5.0)
                    fval = func(x_clip)
                    evals += 1
                    fitvals.append(fval)
                    if fval < best_f:
                        best_f = fval
                        best_x = x_clip.copy()
                    if time.time() > deadline - 5.0:
                        break
                # Sort
                order = np.argsort(fitvals)
                fitvals_sorted = np.array([fitvals[i] for i in order])
                x_sorted = np.array([pops[i] for i in order])

                # Update mean
                xmean_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)

                # Update covariance matrix
                ys = (x_sorted[:mu] - xmean) / sigma
                delta_mean = xmean_new - xmean
                rank_mu = np.outer(delta_mean, delta_mean)
                rank_one = np.outer(ys[:, 0], ys[:, 0]) if dim == 1 else np.outer(ys[0], ys[0])
                # Actually rank_one uses ys[0] which is the first selected
                rank_one = np.outer(ys[0], ys[0])

                C = (1 - c1 - cmu) * C + c1 * (rank_one + (1 - nu) * np.eye(dim)) + cmu * rank_mu
                # Symmetrize
                C = (C + C.T) / 2

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C ** (-0.5)) @ delta_mean / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) <= (1.4 + 2 / (dim + 1)) * chiN
                pc = (1 - cc) * pc + hsig * cc * delta_mean / sigma
                sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)
                xmean = xmean_new

                # Check convergence
                if np.linalg.norm(xmean - xmean_new) < 1e-12 * max(1, np.linalg.norm(xmean)):
                    break
                if sigma < 1e-12:
                    break
                if np.all(np.abs(np.diag(C)) < 1e-12):
                    break

            return evals

        # Restart strategy: multiple restarts with decreasing budget
        # First restart: larger budget, then smaller
        # For 5D, typical CMA-ES needs ~300-500 evals per run
        # We'll do up to 3 restarts

        # Initial point: try origin first (common for Rosenbrock)
        x0 = np.zeros(dim)
        sigma0 = 2.0
        # Give first run most of the budget
        max_evals_1 = min(budget, 300)
        used = cmaes_run(x0, sigma0, max_evals_1)

        if time.time() < deadline - 10.0 and used < budget:
            # Second restart from best found
            x0_2 = best_x.copy()
            sigma0_2 = 0.5
            max_evals_2 = min(budget - used, 150)
            used += cmaes_run(x0_2, sigma0_2, max_evals_2)

        if time.time() < deadline - 10.0 and used < budget:
            # Third restart: slightly different initial point
            x0_3 = rng.uniform(-3, 3, dim)
            sigma0_3 = 1.0
            max_evals_3 = min(budget - used, 100)
            used += cmaes_run(x0_3, sigma0_3, max_evals_3)

        if time.time() < deadline - 10.0 and used < budget:
            # Fourth restart: local refinement around best
            x0_4 = best_x.copy()
            sigma0_4 = 0.1
            max_evals_4 = min(budget - used, 50)
            used += cmaes_run(x0_4, sigma0_4, max_evals_4)

        return best_x, best_f
# EVOLVE-BLOCK-END
