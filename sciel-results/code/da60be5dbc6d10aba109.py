"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and final polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        n = self.dim
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * n  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / (n ** 1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (n ** 1.7 + 18))
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Multiple restarts with different initializations
        for restart in range(5):
            if calls_made >= self.budget or time.time() > deadline - 5:
                break
            
            # Initialize
            if restart == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-3, 3, n)
            
            sigma = 2.0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            pop = [x + sigma * C_sqrt @ rng.standard_normal(n) for _ in range(lambda_)]
            pop = [np.clip(p, -5, 5) for p in pop]
            fvals = np.array([func(p) for p in pop])
            calls_made += lambda_
            
            # Track best
            for i in range(lambda_):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            
            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = [pop[i] for i in order]
            fvals_sorted = fvals[order]
            
            # CMA-ES iterations
            while calls_made < self.budget and time.time() < deadline - 5:
                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * pop_sorted[i] for i in range(mu))
                
                # Update sigma
                sigma_old = sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma_old) * (C_inv_sqrt @ (x - x_old))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 20) - np.sqrt(n)))
                sigma = max(1e-10, min(sigma, 100.0))
                
                # Update C
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x - x_old)
                delta_h = (1 - (1 - c_c) ** 20) if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 20) < (1.4 + 2 / (n + 1)) * np.sqrt(1 + c_c) else 0
                C = (1 + c_1 + c_mu) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu / sigma_old ** 2 * sum(weights[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old) for i in range(mu))
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, 1e10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sqrt matrices
                eigvals, eigvecs = np.linalg.eigh(C)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Generate new population
                pop = [x + sigma * C_sqrt @ rng.standard_normal(n) for _ in range(lambda_)]
                pop = [np.clip(p, -5, 5) for p in pop]
                fvals = np.array([func(p) for p in pop])
                calls_made += lambda_
                
                # Track best
                for i in range(lambda_):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                
                # Sort
                order = np.argsort(fvals)
                pop_sorted = [pop[i] for i in order]
                fvals_sorted = fvals[order]
                
                # Termination criteria
                if np.std([fvals_sorted[i] for i in range(mu)]) < 1e-12:
                    break
                if np.linalg.norm(x - x_old) < 1e-10:
                    break
        
        # Final polish: (1+1)-ES with small steps to refine the best solution
        if calls_made < self.budget and time.time() < deadline - 1 and best_x is not None:
            polish_x = best_x.copy()
            polish_f = best_f
            polish_sigma = 0.1
            polish_calls = 0
            max_polish_calls = min(self.budget - calls_made, 50)
            
            while polish_calls < max_polish_calls and time.time() < deadline - 1:
                step = polish_sigma * rng.standard_normal(n)
                candidate = np.clip(polish_x + step, -5, 5)
                candidate_f = func(candidate)
                polish_calls += 1
                calls_made += 1
                
                if candidate_f < polish_f:
                    polish_x = candidate
                    polish_f = candidate_f
                    polish_sigma *= 1.5
                    if polish_f < best_f:
                        best_f = polish_f
                        best_x = polish_x.copy()
                else:
                    polish_sigma *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
