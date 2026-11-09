"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 300  # Safety margin before 330s kill

        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Number of parents
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Strategy parameters
        cs = (lam / 2.0 + 1.0) / (n + 4.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((lam / 2.0 - 1.0) / (n + 1.0)) - 1.0) + cs
        cc = (4.0 - lam / 4.0) / (n + 4.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + lam / 2.0)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + lam / 2.0))

        # Initial state
        rng = np.random.default_rng(0)
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # Initial evaluation
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample population
            Y = rng.multivariate_normal(np.zeros(n), C, size=lam)
            X = x_mean + sigma * Y
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls += lam

            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()

            # Weighted mean of best mu individuals
            w = weights
            y_w = np.dot(w, X[:mu].T)

            # Update covariance matrix adaptation
            # Rank-one update
            C_inv = np.linalg.inv(C)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 / (1.0 + cc)) ** (2.0 * (calls // lam))) < 
                    (1.4 + 2.0 / (n + 1)) * chi_n)
            hsig = 1.0 if hsig else 0.0

            ps = (1.0 - cs) * ps + hsig * np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv @ y_w) / sigma
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w / sigma

            delta_hsig = (1.0 - hsig) * cc * (2.0 - cc)
            
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C)
            
            # Rank-mu update
            Y_w = np.array([X[i] for i in range(mu)])
            C += cmu * np.sum([w[i] * np.outer(Y_w[i] / sigma, Y_w[i] / sigma) for i in range(mu)], axis=0)
            
            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            # Simple check for positive definiteness, if fails, revert to identity scaled
            try:
                C_inv = np.linalg.inv(C)
                if np.any(np.diag(C_inv) < 0):
                    C = np.eye(n)
                    C_inv = np.eye(n)
            except:
                C = np.eye(n)
                C_inv = np.eye(n)

            # Update mean
            x_mean = x_mean + sigma * y_w

            # Update step size
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1.0))
            
            # Safety margin for time
            if time.time() > deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
