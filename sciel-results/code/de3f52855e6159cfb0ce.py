"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f12 (Bent Cigar) with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lambda_ // 2  # number of parents
        # recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Initial state
        x_mean = np.zeros(d)  # start at origin; minimizer is in [-4,4]^5
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        C_inv_sqrt = np.eye(d)  # C^{-1/2} initially I

        best_x = x_mean.copy()
        best_f = np.inf
        calls_used = 0

        def check_time():
            return time.time() > deadline

        # Evaluate initial center
        fx = func(x_mean)
        calls_used += 1
        best_f = fx
        best_x = x_mean.copy()

        while calls_used < budget and not check_time():
            # Generate population
            pop = []
            fvals = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                # Transform using C^{-1/2} (approximate via Cholesky of C)
                # For efficiency, use eigendecomposition periodically; here use direct
                y = sigma * (C_inv_sqrt @ z)
                xi = np.clip(x_mean + y, -5, 5)
                pop.append(xi)
                fi = func(xi)
                calls_used += 1
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                if check_time():
                    break

            if len(pop) < 2:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            pop_sorted = [pop[i] for i in indices]
            fvals_sorted = [fvals[i] for i in indices]

            # Update mean (recombination)
            x_old = x_mean.copy()
            x_mean_new = np.zeros(d)
            for i in range(mu):
                x_mean_new += w[i] * pop_sorted[i]
            x_mean = x_mean_new

            # Update step size (CSA)
            z_mean = np.zeros(d)
            for i in range(mu):
                z_i = np.linalg.solve(C_inv_sqrt.T, (pop_sorted[i] - x_old) / sigma)
                z_mean += w[i] * z_i
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ z_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_used - int(calls_used / lambda_) + 1))) < (1.4 + 2 / (d + 1)) * chi_n
            if hsig:
                pc = (1 - cc) * pc + cc / damps * (x_mean - x_old) / sigma

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                diff = pop_sorted[i] - x_old
                C += cmu * w[i] * np.outer(diff, diff)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C^{-1/2}
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50.0)

            # Keep mean in bounds
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
