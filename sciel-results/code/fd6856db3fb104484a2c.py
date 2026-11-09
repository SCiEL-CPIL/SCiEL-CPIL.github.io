"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with adaptive population for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 280.0

        d = self.dim
        N = self.budget

        def eval_func(x):
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(x)

        # CMA-ES implementation
        rng = np.random.default_rng(42)

        # Initial point: origin (good default for BBOB)
        mean = np.zeros(d)
        sigma = 2.0  # Initial step size
        C = np.eye(d)  # Covariance matrix
        C_inv = np.eye(d)
        pc = np.zeros(d)  # Evolution path for covariance
        ps = np.zeros(d)  # Evolution path for sigma

        # Parameters
        mu = max(1, d // 2)  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
        cs = (mueff + 2.0) / (d + mueff + 5.0)
        c1 = (2.0 - 1.0 / (d + 1.0) + 2.0 * mueff / d) / ((d + 2.0) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 2.0 / d) / ((d + 2.0) ** 2 + mueff))
        chiN = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

        # Population size: start small, grow as we learn
        lambda_pop = max(mu + 1, min(d + 3, 10))

        best_x = mean.copy()
        best_fx = eval_func(best_x)

        # Track best value for early stopping
        no_improve_counter = 0
        stagnation_threshold = 50

        while self.calls < N and (time.time() - self.start_time) < self.time_limit:
            # Generate population
            pop = np.zeros((lambda_pop, d))
            for i in range(lambda_pop):
                z = rng.standard_normal(d)
                # Cholesky decomposition for sampling from C
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(d)
                pop[i] = mean + sigma * (L @ z)
                # Clip to box
                pop[i] = np.clip(pop[i], -5.0, 5.0)

            # Evaluate population
            fvals = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                try:
                    fvals[i] = eval_func(pop[i])
                except StopIteration:
                    fvals[i] = np.inf
                    break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # Update mean
            mean_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            mean = mean_new

            # Update evolution paths
            dmean = mean_new - mean
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * C_inv @ (dmean / sigma)
            if np.linalg.norm(ps) / chiN < (1.4 + 1.0 / cs) * np.sqrt(1.0 - (1.0 - cs) ** 2):
                pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (dmean / sigma)

            # Update covariance
            outer = np.zeros((d, d))
            for i in range(mu):
                outer += np.outer(pop_sorted[i] - mean_new, pop_sorted[i] - mean_new)
            delta_heav = (1.0 + 2.0 * max(0.0, (np.linalg.norm(ps) / chiN - 1.0))) * cc * (2.0 - cc)
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_heav * C) + cmu / mueff * outer

            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(d)

            # Update sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Stagnation check: if no improvement for many iterations, reduce sigma and try new region
            if no_improve_counter > stagnation_threshold:
                # Reset to best point with larger sigma
                mean = best_x.copy()
                sigma = max(sigma, 1.0)
                C = np.eye(d)
                ps = np.zeros(d)
                pc = np.zeros(d)
                no_improve_counter = 0

        # Final polish with Powell if we have budget left
        if self.calls < N and (time.time() - self.start_time) < self.time_limit:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'xatol': 1e-8,
                        'disp': False
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
