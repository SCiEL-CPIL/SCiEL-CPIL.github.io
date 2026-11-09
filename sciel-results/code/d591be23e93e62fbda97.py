"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with box-constraint handling for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Use a smaller population to allow more generations within the budget
        # Standard lambda is ~8 for dim=5, which gives ~60 generations.
        # lambda=4 gives ~125 generations, allowing better covariance adaptation.
        lam = 4
        mu = lam // 2
        
        # Weights
        # Logarithmic ranking weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = (2 - cc + 2 * mueff / (dim - 1)) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial point: Random within the box to avoid bias towards center
        # The problem states the minimizer is shifted, so center is not guaranteed.
        xmean = rng.uniform(-5, 5, dim)
        sigma = 1.5  # Initial step size, slightly smaller than box half-width

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        Cinv = np.eye(dim)
        D = np.ones(dim)
        B = np.eye(dim)

        # Cache for function evaluations
        best_x = xmean.copy()
        best_f = func(best_x)
        n_evals = 1

        def count_func(x):
            nonlocal n_evals, best_x, best_f
            n_evals += 1
            f = func(x)
            if f < best_f:
                best_x = x.copy()
                best_f = f
            return f

        while n_evals < budget:
            if time.time() > deadline:
                break

            # Generate population
            X = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Sample from N(xmean, sigma^2 * C)
                x = xmean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                X[i] = x

            # Evaluate
            F = np.array([count_func(X[i]) for i in range(lam)])

            # Sort by fitness
            idx = np.argsort(F)
            Xsorted = X[idx]
            Fsorted = F[idx]

            # Update mean
            xnew = np.sum(weights[:, None] * Xsorted[:mu], axis=0)
            xnew = np.clip(xnew, -5, 5)

            # Update sigma with CSA
            y = (xnew - xmean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv @ y)
            
            # Hessian of the step size adaptation
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_evals // lam + 1)))) / \
                   np.sqrt(dim) * (1 + cc) < (1.4 + 2 / (dim + 1)) * (1 - cs)
            
            pc = (1 - cc) * pc + hsig * cc * y
            
            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc * (1 - cc) * C) + \
                cmu * np.sum(weights[:, None, None] * np.outer(Xsorted[:mu] - xmean, Xsorted[:mu] - xmean) / sigma**2, axis=0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                Cinv = (B / D) @ (B / D).T
            except:
                B = np.eye(dim)
                D = np.ones(dim)
                Cinv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            xmean = xnew

        return best_x, best_f
# EVOLVE-BLOCK-END
