"""CMA-ES optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Step size adaptation
        cs = (mueff + 1) / (dim + mueff + 2)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs

        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (1 - cs) / (1 + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Initialization
        rng = np.random.default_rng(42)

        xmean = np.zeros(dim)
        sigma = 1.0

        C = np.eye(dim)

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = xmean.copy()
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial point
        x0 = xmean.copy()
        f0 = evaluate(x0)
        budget_used = 1

        # Main CMA-ES loop
        while budget_used < budget:
            if time.time() > deadline:
                break

            # Check if we have enough budget for a full generation
            if budget - budget_used < lam:
                break

            # Compute Cinv_sqrt
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            Cinv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

            # Sample
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = xmean + sigma * (Cinv_sqrt @ z)
                ys.append(y)

            # Evaluate
            fvals = np.array([evaluate(y) for y in ys])
            budget_used += lam

            # Sort
            idx = np.argsort(fvals)
            ys = np.array(ys)[idx]
            fvals = fvals[idx]

            # Update xmean
            xmean_new = xmean + sigma * sum(weights[i] * ys[i] for i in range(mu))

            # Update p_sigma
            d = (xmean_new - xmean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ d)

            # Update sigma
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update C
            y = (xmean_new - xmean) / sigma
            rank1 = np.outer(y, y)

            y_selected = (ys[:mu] - xmean) / sigma
            rankmu = sum(weights[i] * np.outer(y_selected[i], y_selected[i]) for i in range(mu))

            C = (1 - c1 * (1 - cs)**2) * C + c1 * rank1 + cmu * rankmu
            C = (C + C.T) / 2

            xmean = xmean_new

        # Polish with Nelder-Mead if budget allows
        if budget - budget_used > 10 and time.time() < deadline:
            def nm_func(x):
                return evaluate(x)

            try:
                result = minimize(nm_func, best_x, method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200})
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
