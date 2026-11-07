"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        # CMA-ES Parameters
        # Standard population size for dim=5
        # lambda = 4 + floor(3 * log(mu)) is common, but let's use fixed standard values
        # For dim 5, lambda=9 is a good standard choice
        lam = 4 + int(3 * np.log(1 + dim))  # 4 + 3*1.609 = 8.8 -> 8? No, 4+3*log(6) ~ 8.8. Let's use 9.
        lam = 9
        mu = lam // 2  # 4
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cs

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, box is [-5,5]
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initial population
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
            return fx

        # Evaluate initial mean
        f_mean = eval_point(x_mean)

        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lam, dim))
            fvals = np.zeros(lam)
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            for i in range(lam):
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * (L @ z)
                fvals[i] = eval_point(pop[i])
                
                if calls_used >= budget:
                    break

            if calls_used >= budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(fvals)
            pop = pop[sorted_idx]
            fvals = fvals[sorted_idx]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)

            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (L @ (x_mean - x_mean_old))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n) < (1.4 + 2 / (dim + 1))
            
            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old) / sigma

            # Update covariance matrix
            delta_heav = (1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) - cc) * (1 - np.linalg.norm(pc)**2 / (dim * (1 - (1 - cc)**2)))
            
            # Rank-one update
            rank1 = np.outer(pc, pc)
            
            # Rank-mu update
            diff_pop = (pop[:mu] - x_mean_old) / sigma
            rankmu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(diff_pop, diff_pop).transpose(0, 2, 1), axis=0)

            C = (1 - (c1 + cmu)) * C + c1 * rank1 + cmu * rankmu

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Simple regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)

            # Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Convergence check
            if sigma < 1e-12:
                break
            
            # If best point hasn't improved for a while, might want to restart or perturb
            # But CMA-ES is robust.
            
            # Optional: If we are stuck, we can try a random restart or increase sigma
            # For now, standard CMA-ES logic.

        return best_x, best_f
# EVOLVE-BLOCK-END
