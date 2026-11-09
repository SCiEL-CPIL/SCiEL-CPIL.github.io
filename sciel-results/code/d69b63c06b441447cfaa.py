"""Rewritten optimizer using CMA-ES for Rastrigin rotated f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with large population for global diversity on rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For 5D, lambda = 4 + floor(3 * log(n)) ≈ 4 + 3*1.6 = 8, but we use larger for diversity
        lam = 10  # larger population for global exploration
        mu = lam // 2  # number of parents

        # Initial mean: center of the box
        x_mean = np.zeros(n)
        
        # Initial sigma: scale to explore the box
        # Box is [-5, 5]^5, so range is 10. Use a reasonable initial step size.
        sigma = 2.0

        # Covariance matrix (identity initially)
        C = np.eye(n)
        
        # Step sizes for covariance adaptation
        cs = (mu + 2) / (n + mu + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + cs
        cc = (4 + n / mu) / (n + 4)
        cmu1 = 2 / ((n + 1.3) ** 2 + mu)
        cmu2 = min(1 - cmu1, 2 * (mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Memory
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        # Evaluate initial point
        x_eval = np.clip(x_mean, -5, 5)
        f_best = func(x_eval)
        best_x = x_eval.copy()
        best_f = f_best
        calls_made = 1

        # Pre-generate some random points for initial population diversity?
        # No, CMA-ES generates them from the distribution.

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda offspring
            xs = []
            for i in range(lam):
                # Sample from multivariate normal
                # x = x_mean + sigma * z
                # z = C^{1/2} * N(0, I)
                # Use Cholesky decomposition for C^{1/2}
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    L = np.eye(n)
                
                z = L @ np.random.standard_normal(n)
                x_i = x_mean + sigma * z
                x_i = np.clip(x_i, -5, 5)
                xs.append(x_i)
            
            # Evaluate offspring
            fs = []
            for x_i in xs:
                if calls_made >= budget:
                    break
                f_i = func(x_i)
                fs.append(f_i)
                calls_made += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(fs) < lam:
                # Not enough evaluations left to complete a generation
                break

            fs = np.array(fs)
            xs = np.array(xs)

            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]

            # Take top mu
            xs_parents = xs_sorted[:mu]
            
            # Recombination
            w = weights
            x_new = np.sum(w[:, None] * xs_parents, axis=0)

            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (x_new - x_mean))

            # Hessian correction
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (calls_made / (cs * n))) < (1.4 + 2 / (n + 1)) * chi_n)
            hsig = 1.0 if hsig else 0.0

            # Update p_c
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * (x_new - x_mean)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmu1 + cmu2 - cmu1 - delta_hsig) * C
            C -= cmu2 * np.outer(p_c, p_c)
            
            for i in range(mu):
                y_i = (xs_parents[i] - x_mean) / sigma
                C += cmu1 * weights[i] * np.outer(y_i, y_i)

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_mean = x_new

            # Ensure C remains positive definite (numerical stability)
            # Symmetrize
            C = (C + C.T) / 2

            # If sigma becomes too small or too large, handle it
            if sigma < 1e-10:
                break
            if sigma > 50:
                sigma = 50

            # Check for convergence
            if np.linalg.norm(C) < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
