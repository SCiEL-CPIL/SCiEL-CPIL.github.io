# EVOLVE-BLOCK-START
"""
CMA-ES Optimizer for BBOB Sphere (f1).

Strategy:
1. Use Covariance Matrix Adaptation (CMA-ES), a state-of-the-art derivative-free
   evolutionary strategy that adapts both the step size and the shape of the
   search distribution (via a covariance matrix) to the local landscape.
2. CMA-ES is particularly effective on smooth, convex, and symmetric functions
   like Sphere, as it can quickly align the search direction with the principal
   axes of the function's level sets.
3. Implement early termination when the covariance matrix becomes ill-conditioned
   (condition number > 1e12) or the step size becomes negligible, to avoid
   wasting budget on redundant evaluations.
4. Use a robust initialization and adaptive population size to balance exploration
   and exploitation within the 500-call budget.

This keeps the same entry point: Optimizer(budget, dim) and __call__(func).
Returns (best_x, best_f).
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        mu = lam // 2  # Number of recombination vectors
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)  # Effective population size

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initialization
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        best_f = f0
        best_x = x0.copy()

        sigma = 1.0  # Initial step size
        mean = x0.copy()
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)  # Inverse covariance matrix
        C_sqrt = np.eye(dim)  # Square root of covariance matrix
        pc = np.zeros(dim)  # Evolution path for covariance adaptation
        ps = np.zeros(dim)  # Evolution path for step size adaptation
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Check for convergence/termination
            # Condition number of C
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] > 0:
                cond = eigvals[-1] / eigvals[0]
                if cond > 1e12:
                    break
            if sigma < 1e-12:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition for C_sqrt
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, use identity
                    C_sqrt = np.eye(dim)
                
                z = np.random.randn(dim)
                y = C_sqrt @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls += 1
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                ys.append(y)
                xs.append(x)
                fs.append(f)

            if len(fs) < lam:
                break

            # Sort by fitness
            sort_idx = np.argsort(fs)
            ys = [ys[i] for i in sort_idx]
            xs = [xs[i] for i in sort_idx]
            fs = [fs[i] for i in sort_idx]

            # Update mean
            mean_old = mean.copy()
            mean = sum(w[i] * xs[i] for i in range(mu))

            # Update step size (sigma)
            y_mean = sum(w[i] * ys[i] for i in range(mu))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt @ y_mean)
            norm_ps = np.linalg.norm(ps)
            h_sig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Update covariance matrix
            h_c = 1 if norm_ps / np.sqrt(1 - (1 - c_c)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            pc = (1 - c_c) * pc + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
            delta_h = (1 - h_c) * (1 - (1 - c_c)**lam)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * sum(w[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            C = (C + C.T) / 2  # Ensure symmetry

            # Update inverse and square root
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
