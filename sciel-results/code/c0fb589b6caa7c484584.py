"""Rewritten optimizer for BBOB f23 (Katsuura) using CMA-ES style adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style adaptive search for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Start at origin (Katsuura minimum is often near 0 after transformation)
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = 4 * (int(np.log(d)) + 1)  # population size
        # Initial step size
        sigma = 1.0

        # Covariance matrix (start as identity)
        C = np.eye(d)
        C_inv = np.eye(d)
        C_sqrt = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Update rates
        cs = (lambda_ + 2) / (d + lambda_ + 5)
        cmu = min(1 - 2 * cs, 2 / (d + lambda_ + 2))
        cmu = cmu / (d * lambda_)  # per-individual weight

        # Eigenvalue decay for C
        c1 = (2 + 2 / (d + 1)) / (d**1.8 + lambda_ + 2)
        cmu_ = 2 * (cmu - c1)

        # Step size adaptation (CCSA)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)

        # Main loop
        calls = 1  # already made one call for initial point
        generation = 0

        while calls < B:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                # Sample from multivariate normal
                z = rng.standard_normal(d)
                # Transform using C_sqrt
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])

            # Update mean
            w = weights
            x_new = np.sum(w[:, None] * np.array([o[0] for o in offspring[:mu]]), axis=0)

            # Update evolution paths
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv @ (C_sqrt.T @ (x_new - x) / sigma)

            # pc update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (generation + 1))) < (1.4 + 1 / d) * chi_d:
                pc = (1 - cmu) * pc + cmu * (C_inv @ (x_new - x) / sigma)

            # Update covariance matrix
            delta_d = (1 - cmu) * (1 + c1 / c1) + c1 * np.trace(C)
            C = (1 + c1 * (1 + cmu - c1) - c1) * C + c1 * (pc @ pc.T - C)
            for i in range(mu):
                di = x_new - x  # This is wrong, should be individual
                # Actually we need the individual differences
            # Let me redo this more carefully

            # Recompute C update properly
            C_old = C.copy()
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 if np.linalg.norm(ps) / chi_d > 1.4 + 1 / d else 0) * C)
            for i in range(mu):
                di = np.array(offspring[i][0]) - x
                C += cmu * (di @ di.T)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(d)

            # Recompute C_inv and C_sqrt
            try:
                C_inv = np.linalg.inv(C)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                C = np.eye(d)
                C_inv = np.eye(d)
                C_sqrt = np.eye(d)

            # Update sigma (CCSA)
            sigma *= np.exp(cs / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            x = x_new
            generation += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
