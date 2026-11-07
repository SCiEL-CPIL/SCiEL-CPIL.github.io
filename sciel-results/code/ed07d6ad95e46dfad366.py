import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 315s to allow for overhead
        deadline = start_time + 315.0
        calls = 0
        max_calls = self.budget
        best_x = None
        best_f = np.inf
        dim = self.dim
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        # lambda = 10 is good for dim=5
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        m = np.zeros(dim)
        C = np.eye(dim)
        sigma = 1.5 # Slightly larger to cover [-5,5] better initially
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        gen = 0
        no_improve_count = 0
        stagnation_threshold = 20 # Restart if no improvement for 20 gens
        restarts = 0
        max_restarts = 1 # Only allow 1 restart to save budget for refinement

        # Helper to sample from N(m, sigma^2 C)
        def sample_cma(m, sigma, C, n_samples):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                # Note: L here is symmetric sqrt, not triangular. 
                # x = m + sigma * L @ z works if L is sqrt(C).
                # For Cholesky, L is lower triangular such that L @ L.T = C.
                # z @ L.T has covariance L @ L.T = C.
                # z @ L has covariance L.T @ L = C.
                # So both work for sampling if L is a valid matrix square root.
            
            z = rng.standard_normal((n_samples, dim))
            # x = m + sigma * (z @ L.T)
            samples = m + sigma * (z @ L.T)
            return samples

        # Main Loop
        while calls < max_calls and time.time() < deadline:
            # 1. Generate population
            if calls + lambda_ > max_calls:
                # If we don't have enough budget for a full generation,
                # just evaluate as many as we can? 
                # CMA-ES needs full generations for update. 
                # If we can't do a full generation, break.
                break
            
            pop = sample_cma(m, sigma, C, lambda_)
            pop_fitness = np.full(lambda_, np.inf)
            
            for i in range(lambda_):
                try:
                    pop_fitness[i] = wrapper(pop[i])
                except (ValueError, TimeoutError):
                    break
            
            if np.any(np.isinf(pop_fitness)):
                break

            # 2. Sort population by fitness
            idx = np.argsort(pop_fitness)
            pop = pop[idx]
            pop_fitness = pop_fitness[idx]
            
            # Check for improvement
            if pop_fitness[0] < best_f - 1e-12:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 3. Update CMA-ES parameters
            # Select best mu individuals
            y_best = pop[:mu]
            
            # Compute weighted mean (new mean)
            m_old = m
            m = np.sum(weights[:, np.newaxis] * y_best, axis=0)
            
            # Update step size control (CSA)
            # Compute C^{-1/2}
            try:
                L = np.linalg.cholesky(C)
                # C^{-1/2} = L^{-T} L^{-1}
                # Or use eigendecomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            delta_m = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_m)
            
            # h_sigma condition
            phi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(dim+1)) * phi else 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / (dim + 2) * (np.linalg.norm(p_sigma) / phi - 1))
            # Standard formula is exp(c_sigma * (||p_sigma||/phi - 1))
            # Let's stick to the standard one
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / phi - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ delta_m)
            
            # Update covariance matrix
            rank_one = np.outer(p_sigma, p_sigma)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_inv_sqrt @ (y_best[i] - m_old) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2
            
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            gen += 1
            
            # 4. Check for stagnation and restart
            if no_improve_count > stagnation_threshold and restarts < max_restarts:
                # Restart: center on best point, increase sigma
                m = best_x.copy()
                sigma = max(sigma * 2.0, 0.5)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                restarts += 1

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
