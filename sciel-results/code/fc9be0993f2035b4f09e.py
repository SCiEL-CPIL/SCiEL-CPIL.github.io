"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        rng = np.random.default_rng(42)

        # Initial point: center of box
        x = np.zeros(dim)
        sigma = 3.0  # Initial step size (box is [-5,5], so 3 is reasonable)

        # CMA-ES parameters
        lambda_ = 4 * (dim + 1)  # Population size
        mu = lambda_ // 2        # Number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        B = np.eye(dim)
        D = np.ones(dim)

        # Evaluate initial point
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            fitnesses = []

            for i in range(lambda_):
                # Sample new point
                y = x + sigma * B @ (D * rng.standard_normal(dim))

                # Clip to box
                y = np.clip(y, -5, 5)

                # Evaluate
                fy = func(y)
                calls_made += 1

                offspring.append(y)
                fitnesses.append(fy)

                # Update best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                # Check deadline mid-population
                if time.time() > deadline:
                    break

            if len(offspring) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fitnesses)
            offspring = [offspring[i] for i in idx]
            fitnesses = [fitnesses[i] for i in idx]

            # Compute weighted mean of best mu
            wmean = np.zeros(dim)
            for i in range(mu):
                wmean += weights[i] * offspring[i]

            # Update evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * B @ (D * (wmean - x))

            # Check sigma update condition
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lambda_))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-8, 100)

            # Update evolution path for covariance
            xmean_old = x
            x = wmean

            # Only update covariance if we have enough points
            if len(offspring) >= mu:
                yk = [(offspring[i] - xmean_old) / sigma for i in range(mu)]
                delta = 0.0
                for i in range(mu):
                    delta += weights[i] * (np.linalg.norm(yk[i]) ** 2)
                if delta < 1e-10:
                    delta = 1e-10

                # Update pc
                pc = (1 - c_c) * pc
                for i in range(mu):
                    pc += weights[i] * np.outer(yk[i], yk[i])

                # Update covariance matrix
                C = B @ np.diag(D ** 2) @ B.T
                C = (1 + c_1 + c_mu) * C
                C -= (c_1 + c_mu) * np.eye(dim)
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(yk[i], yk[i])
                C += c_1 * (pc + pc.T - 2 * c_c * pc)

                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-10)
                D = np.sqrt(eigvals)
                B = eigvecs

        return best_x, best_fx
# EVOLVE-BLOCK-END
