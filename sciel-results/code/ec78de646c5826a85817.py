"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB-style Sphere f1 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard
        t0 = time.time()
        deadline = t0 + 280.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Selection size mu
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        # Variance-effectiveness
        mueff = 1.0 / np.sum(w ** 2)

        # Initial parameters
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + mueff / dim) / ((dim + 4) * (2 + mueff / dim))
        cs = (mueff + 2) / (dim + mueff + 2)
        cmu = min(1 - cc, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 - 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1) - 1) + cs

        # Cholesky decomposition of C
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(dim)

        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)

                # Update best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                # Check time inside loop for safety
                if time.time() > deadline:
                    break

            # If we ran out of time, break
            if len(offspring) == 0:
                break

            # Sort offspring
            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]

            # Update mean
            x_new = np.sum(w[:len(offspring)], keepdims=True).T * offspring[:len(offspring)]
            
            # Update evolution path for sigma (ps)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(B.T) * (B.T @ (x_new - x) / sigma)
            
            # Update evolution path for covariance (pc)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) <
                    (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (calls + 1)))
            pc = (1 - cc) * pc + hsig * cc * (x_new - x) / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmu * delta_hsig - cs * (2 - cs)) * C
            C += cmu * np.sum(w[:len(offspring)] * np.outer(offspring[:len(offspring)] - x, 
                     offspring[:len(offspring)] - x), axis=0) / sigma ** 2
            C += (cmu * delta_hsig - cc * (2 - cc) + cc ** 2) * np.outer(pc, pc)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except:
                B = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) * (1 + 1 / (2 * dim)) - 1))
            sigma = max(sigma, 1e-12)

            # Update mean
            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
