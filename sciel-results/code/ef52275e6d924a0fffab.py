"""CMA-ES optimizer for BBOB f12 (Bent Cigar)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda = population size, choose based on dim
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Number of iterations we can afford
        # Each iteration uses lam evaluations
        # We have budget evaluations total
        # Start with 1 evaluation for initial point
        max_iters = (budget - 1) // lam
        if max_iters < 1:
            max_iters = 1

        # Initialize
        x_mean = rng.uniform(-5, 5, dim)
        f_mean = func(x_mean)
        budget_used = 1

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # CMA-ES parameters (standard)
        cmu = 0  # rank-mu weight (0 for simplicity, use rank-one only)
        # Actually let's use standard CMA-ES with rank-mu
        # Weights for top mu individuals
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Update rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = (mueff - 2 + c1) / ((dim + 2) ** 2 + mueff)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        # damps for step size adaptation
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Precompute sqrt of C for sampling
        def update_sqrt_C(C):
            # Use Cholesky or eigendecomposition
            # For stability, use eigendecomposition
            evals, evecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            evals = np.maximum(evals, 1e-12)
            sqrt_C = evecs @ np.diag(np.sqrt(evals))
            return sqrt_C

        sqrt_C = update_sqrt_C(C)

        best_x = x_mean.copy()
        best_f = f_mean

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if budget_used >= budget:
                break

            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (sqrt_C @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)

            # Evaluate
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)
                budget_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                if budget_used >= budget:
                    break

            if len(fs) < lam:
                # Pad if we ran out of budget
                while len(fs) < lam:
                    fs.append(fs[0] if fs else 0.0)
                    ys.append(ys[0] if ys else np.zeros(dim))

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * np.array(ys[:mu]), axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update covariance
            diff = x_mean - x_old
            C_old = C.copy()
            C = (1 - c1 - cmu) * C + c1 * np.outer(diff, diff)
            for i in range(mu):
                yi = ys[i] - x_old
                C += cmu * weights[i] * np.outer(yi, yi)

            # Ensure positive semi-definite
            # Regularize
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(dim)

            # Update path accumulators
            inv_sqrt_C = np.linalg.inv(sqrt_C)
            pc = (1 - cc) * pc + cc * inv_sqrt_C @ diff / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C @ diff / sigma

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update sqrt_C
            sqrt_C = update_sqrt_C(C)

        return best_x, best_f
