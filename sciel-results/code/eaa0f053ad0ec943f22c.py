"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # CMA-ES parameters
        sigma0 = 2.0
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # CMA-ES state
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = sigma0

        def evaluate(x):
            nonlocal call_count, best_x, best_fx
            x_clipped = np.clip(x, -5, 5)
            fval = func(x_clipped)
            call_count += 1
            if fval < best_fx:
                best_fx = fval
                best_x = x_clipped.copy()
            return fval

        # Main loop
        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                offspring[i] = x + sigma * (B @ (D * z))

            # Evaluate
            fitness = np.zeros(lam)
            for i in range(lam):
                fitness[i] = evaluate(offspring[i])

            # Sort by fitness
            order = np.argsort(fitness)
            mu = lam // 2  # number of parents
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = w.sum() ** 2

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[order[i]]
            x = x_new

            # Update evolution paths
            mean_old = x - (x_new - x)  # this is just x_old, but we need the old mean
            # Actually, let's track the old mean properly
            # Recompute: the mean before update was the old x
            # Let's redo this more carefully

            # Actually, I need to restructure. Let me track old_mean
            # For now, let's just proceed with the standard CMA-ES update

            # Update step size (sigma) using CSA
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (B @ (D @ ((x - (x - sigma * B @ (D @ np.random.randn(dim)))) / sigma)))
            # This is getting complicated. Let me simplify.

            # Let me restart with a cleaner implementation

            break  # This approach is getting messy, let me use a different strategy

        # Simpler approach: use a well-tested CMA-ES implementation
        # Let me rewrite from scratch with a clean CMA-ES

        # Reset
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1
        deadline = time.time() + 290.0

        sigma = 2.0
        lam = max(4, int(4 + 3 * np.log(dim)))
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9.2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        # Eigendecomposition of C
        D = np.ones(dim)
        B = np.eye(dim)

        pc = np.zeros(dim)
        ps = np.zeros(dim)

        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = w.sum() ** 2

        def evaluate(x):
            nonlocal call_count, best_x, best_fx
            x_clipped = np.clip(x, -5, 5)
            fval = func(x_clipped)
            call_count += 1
            if fval < best_fx:
                best_fx = fval
                best_x = x_clipped.copy()
            return fval

        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                # Transform z using B and D
                transformed_z = B @ (D * z)
                offspring[i] = x + sigma * transformed_z

            # Evaluate offspring
            fitness = np.zeros(lam)
            for i in range(lam):
                fitness[i] = evaluate(offspring[i])

            # Sort
            order = np.argsort(fitness)
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * (offspring[order[i]] - x) / sigma

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * y_w
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * y_w

            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count - 1) / (cs * dim)))) < (1.4 + 2 / (dim + 1)) * chi_n
            delta = 1.0 if hsig else 0.0

            C = (1 + c1 * delta * mueff + cmu - c1) * C
            C -= c1 * np.outer(pc, pc)
            for i in range(mu):
                yi = (offspring[order[i]] - x) / sigma
                C += cmu * w[i] * np.outer(yi, yi)

            # Ensure positive definiteness
            C = 0.5 * (C + C.T)

            # Update step size
            sigma *= np.exp(cs / (2 * ds) * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = x + y_w * sigma

            # Update B and D via eigendecomposition
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-20))
            B = B2
            D = D2

        return best_x, best_fx
# EVOLVE-BLOCK-END
