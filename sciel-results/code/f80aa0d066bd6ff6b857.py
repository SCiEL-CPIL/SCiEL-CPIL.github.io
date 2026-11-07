"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with adaptive restart for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        def cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with given budget."""
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 10:
                return

            dim = self.dim
            lam = 4 + int(3 * np.log(dim + 1))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) * (dim + 1) + 2)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) * (dim + mueff)))

            sigma = sigma0
            xmean = np.array(x0, dtype=float)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            gen_count = 0
            no_improve = 0
            local_best = best_f

            # Evaluate initial point
            if calls_made < self.budget and time.time() < deadline - 5:
                try:
                    fx = counted_func(xmean)
                    if fx < local_best:
                        local_best = fx
                except RuntimeError:
                    return

            while max_evals > 0 and calls_made < self.budget:
                if time.time() > deadline - 5:
                    break

                # Generate samples
                samples = []
                for i in range(lam):
                    if calls_made + len(samples) >= self.budget or time.time() > deadline - 3:
                        break
                    z = rng.standard_normal(dim)
                    xi = xmean + sigma * (B @ (D * z))
                    xi = np.clip(xi, -5, 5)
                    samples.append(xi)

                if len(samples) < 2:
                    break

                # Evaluate samples
                fvals = []
                for xi in samples:
                    if calls_made >= self.budget or time.time() > deadline - 3:
                        break
                    try:
                        fi = counted_func(xi)
                        fvals.append(fi)
                    except RuntimeError:
                        break

                if len(fvals) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                x_sorted = [samples[i] for i in idx[:mu]]
                f_sorted = [fvals[i] for i in idx[:mu]]

                # Check improvement
                if f_sorted[0] < local_best:
                    local_best = f_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stop if no improvement for many generations
                if no_improve > 30:
                    break

                # Update mean
                xnew = np.sum([w[i] * x_sorted[i] for i in range(mu)], axis=0)

                # Update evolution paths
                diff = (xnew - xmean) / (sigma + 1e-15)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D * diff))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_count + 1)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(dim)
                if hsig:
                    pc = (1 - c1) * pc + c1 * diff
                else:
                    pc = (1 - c1) * pc

                # Update covariance
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (delta_hsig - c1 + c1 * c1) * np.eye(dim)) + \
                    cmu * np.sum([w[i] * np.outer(x_sorted[i] - xmean, x_sorted[i] - xmean)
                                  for i in range(mu)], axis=0)

                # Regularization and re-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

                xmean = xnew
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-12, 1e8)

                gen_count += 1
                max_evals = self.budget - calls_made

        # Phase 1: Multi-start CMA-ES with increasing sigma
        # Start with large sigma to explore, then refine
        n_restarts = 3
        sigmas = [3.0, 1.5, 0.5]

        for i in range(n_restarts):
            if time.time() > deadline - 40:
                break

            x0 = rng.uniform(-4, 4, self.dim)
            sigma0 = sigmas[i] if i < len(sigmas) else 0.2

            # Allocate budget: more for later (finer) runs
            if i == 0:
                max_evals = 120
            elif i == 1:
                max_evals = 150
            else:
                max_evals = 180

            cma_es(x0, sigma0, max_evals)

        # Phase 2: Final polish with very small sigma on best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 10:
            remaining = self.budget - calls_made
            if remaining > 20:
                cma_es(best_x, 0.05, min(remaining, 80))

        # Phase 3: If still budget left, do a few random restarts with tiny sigma
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            for _ in range(5):
                if calls_made >= self.budget or time.time() > deadline - 3:
                    break
                # Small perturbation around best
                x0 = best_x + rng.normal(0, 0.01, self.dim)
                x0 = np.clip(x0, -5, 5)
                try:
                    f0 = counted_func(x0)
                except RuntimeError:
                    break

        # Fallback: if no calls made, do one
        if calls_made == 0:
            x0 = rng.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
