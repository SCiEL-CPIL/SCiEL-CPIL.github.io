"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(4, int(round(4 + 3 * np.log(d))))

        # Initial point
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Initial step size
        sigma = 3.0

        # Covariance matrix
        C = np.eye(d)
        
        # Parameters
        cs = (lam // 2 + 1) / (d + lam)
        ds = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1)) - 1) + cs
        cc = (4 + d / lam) / (d + 4 + 2 * lam / (d + 1))
        cmu = min(1 - cc, 2 * (d - 2 + 2 / (d + 1)) / ((d + 2) * (lam // 2 + 1)))
        chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Weights
        mu = lam // 2
        raw_weights = np.array([np.log(mu + 1) - np.log(i + 1) for i in range(mu)] + [0] * (lam - mu))
        weights = raw_weights / raw_weights.sum()
        mueff = 1 / np.sum(weights[:mu] ** 2)

        gen = 0
        while calls + lam <= budget:
            gen += 1
            # Generate population
            zs = rng.standard_normal((lam, d))
            Y = x + sigma * zs
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lam
            
            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Weighted mean of selected points
            y_mean = np.sum(weights[:mu, None] * Y[:mu], axis=0)
            
            # Update mean
            x = x + y_mean
            
            # p_sigma
            if sigma > 1e-10:
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (
                    np.linalg.norm(C) * y_mean / (sigma * chi)
                )
            else:
                p_sigma = np.zeros(d)
            
            # h_sigma: use a simple threshold
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < 1.4 * chi else 0.0
            
            # p_c
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * y_mean
            
            # Covariance
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            Yw = Y[:mu] * np.sqrt(weights[:mu, None])
            C = (
                (1 + 2 * cmu - delta_h) * C
                + np.dot(Yw.T, Yw)
                - delta_h * np.outer(p_c, p_c)
            )
            C = (C + C.T) / 2
            
            # Sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = max(1e-8, min(sigma, 10.0))
            
            # Convergence check
            if sigma < 1e-5:
                break

        # Use remaining budget for local refinement
        while calls < budget:
            # Try a few random perturbations
            for _ in range(min(3, budget - calls)):
                y = best_x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
