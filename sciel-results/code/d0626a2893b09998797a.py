"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with diversity-aware restarts for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Initial evaluation at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        best_f = f0
        best_x = x0.copy()

        # Multi-start CMA-ES strategy
        # We'll use multiple independent CMA-ES runs with different starting points
        # and restart when progress stalls

        lambda_ = 10  # population size
        mu = 5  # number of parents
        
        # Pre-compute weights
        w = np.zeros(mu)
        w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # CMA-ES parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 - mueff))
        dsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 3))

        # Generate diverse starting points
        start_points = [np.zeros(dim)]
        for _ in range(4):
            sp = rng.uniform(-3, 3, size=dim)
            start_points.append(sp)

        # CMA-ES run with restarts
        for start_idx, x_start in enumerate(start_points):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize CMA-ES state
            x_mean = x_start.copy()
            sigma = 1.0  # larger initial step for exploration
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_c = np.zeros(dim)
            p_sigma = np.zeros(dim)
            hsig = (1 + 1.4 / dim + 2 / ((dim + 1) ** 2)) * np.sqrt((dim + 1) / (dim - 1))
            
            # Evaluate initial point if not already done
            if start_idx > 0:
                fx = func(x_mean)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_mean.copy()
            
            stagnation_count = 0
            local_best_f = best_f
            max_stagnation = 30  # restart after 30 generations without improvement

            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = np.zeros((lambda_, dim))
                fvals = np.zeros(lambda_)
                
                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    if calls >= budget:
                        break
                    
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    pop[i] = y
                    fvals[i] = func(y)
                    calls += 1
                
                if calls >= budget:
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]
                
                # Update best
                gen_best = fvals_sorted[0]
                if gen_best < best_f:
                    best_f = gen_best
                    best_x = pop_sorted[0].copy()
                
                # Check for stagnation
                if gen_best < local_best_f - 1e-8:
                    local_best_f = gen_best
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Restart if stagnating
                if stagnation_count >= max_stagnation:
                    break
                
                # Compute weighted mean
                x_new = np.sum(w[:, np.newaxis] * pop_sorted[:mu], axis=0)
                
                # Update covariance matrix
                y_vec = (x_new - x_mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) - (1 - cc) * C) + \
                    cmu * np.sum(w[:, np.newaxis] * np.outer(pop_sorted[:mu] - x_mean, pop_sorted[:mu] - x_mean) / sigma**2, axis=0)
                C = (C + C.T) / 2
                
                # Update path variables
                p_c = (1 - cc) * p_c + cc * y_vec
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * np.linalg.norm(y_vec)
                
                # Update sigma
                sigma *= np.exp(cs / dsigma * (np.linalg.norm(p_sigma) / hsig - 1))
                
                # Update x_mean
                x_mean = x_new
                x_mean = np.clip(x_mean, -5, 5)
                
                # Recompute C_sqrt periodically
                if calls % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-10, None)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_sqrt = np.eye(dim)
                        C = np.eye(dim)
                
                # Keep sigma in bounds
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # If sigma is too small, restart
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
