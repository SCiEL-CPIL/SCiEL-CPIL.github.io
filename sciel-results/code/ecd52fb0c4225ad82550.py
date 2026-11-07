"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with strict budget control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        c1 = (2 / ((dim + 1.3)**2 + mueff))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initialize population
        pop = []
        for _ in range(lambda_):
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            pop.append((fx, x))
        pop.sort(key=lambda t: t[0])
        x = pop[0][1].copy()
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        sqrtC = B @ np.diag(D) @ B.T
        inv_sqrtC = B @ np.diag(1.0/D) @ B.T

        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline - 5:
                break

            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (sqrtC @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            # Update mean
            new_x = sum(w[i] * offspring[i][1] for i in range(mu))
            new_x = np.clip(new_x, -5, 5)

            # Update step size control
            diff = new_x - x
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrtC @ diff) / sigma

            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*(calls_used - lambda_)))) \
                   < (1.4 + 2/(dim+1)) * chi
            if hsig:
                pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * diff / sigma

            # Update covariance
            delta_h = (1 + np.sign(np.linalg.norm(ps)/chi - 1) * (1 - np.sqrt(1 - cs)) - 1 + cs) \
                      if hsig else 0
            C = (1 - c1 * (1 + (1 + cmu) * delta_h)) * C + c1 * np.outer(pc, pc)
            for i in range(mu):
                zi = inv_sqrtC @ (offspring[i][1] - x) / sigma
                C += cmu * w[i] * np.outer(zi, zi)

            # Update inverse and sqrt
            C += 1e-12 * np.eye(dim)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            sqrtC = B @ np.diag(D) @ B.T
            inv_sqrtC = B @ np.diag(1.0/D) @ B.T

            # Update sigma
            sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            x = new_x

            # Check convergence
            if sigma < 1e-10 or np.max(D) / np.min(D) > 1e14:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
