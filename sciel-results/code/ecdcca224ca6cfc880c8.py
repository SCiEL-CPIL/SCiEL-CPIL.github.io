"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for Katsuura-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(n))  # Population size
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= w.sum()
        mu = lam // 2  # Number of selected parents
        mu_eff = (w[:mu].sum()) ** 2 / (w[:mu] ** 2).sum()

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (np.sqrt(n) + 1) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + np.sqrt(2 * (np.sqrt(n) - 1) ** 2)) / ((n + 2) ** 2 + mu_eff))

        # Initial state
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_x = x_mean
        best_f = func(best_x)
        
        # Precompute square root of C for sampling
        # Using Cholesky decomposition or SVD. SVD is more stable if C becomes singular.
        # For small n, Cholesky is fast. We'll use SVD for robustness.
        
        def sample_from_cma():
            # Sample from N(0, C)
            # C = V S V^T, so sqrt(C) = V sqrt(S)
            # We update C, so we need to recompute sqrt every step or use a trick.
            # Standard CMA-ES uses the update of C directly and samples via Cholesky of C.
            # However, C might not be perfectly PSD due to numerical errors.
            # We will use np.linalg.cholesky and fallback to SVD.
            try:
                L = np.linalg.cholesky(C)
                z = rng.standard_normal(n)
                return x_mean + sigma * L @ z
            except np.linalg.LinAlgError:
                # Fallback to SVD
                U, s, Vt = np.linalg.svd(C)
                L = (U * np.sqrt(s)) @ Vt
                z = rng.standard_normal(n)
                return x_mean + sigma * L @ z

        remaining = budget - 1
        gen = 0
        
        while remaining > 0 and time.time() < deadline:
            # Check time inside loop for safety
            if time.time() > deadline - 5:
                break
            
            # Determine how many samples we can afford this generation
            # Usually 1 generation = lam evaluations
            # If remaining < lam, we can do a partial generation or just stop
            if remaining < lam:
                # Do as many as we can, but CMA-ES needs full population for update
                # If we can't do a full population, it's better to just do random restarts or simple local search?
                # For simplicity, if remaining < lam, we break out of CMA-ES loop 
                # and do a final local polish with (1+1)-ES if time/budget permits.
                break
            
            # Generate population
            offspring = []
            fitnesses = []
            for _ in range(lam):
                if time.time() > deadline - 2:
                    break
                y = sample_from_cma()
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                fitnesses.append(fy)
                remaining -= 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(offspring) < lam:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            offspring = offspring[indices]
            fitnesses = fitnesses[indices]

            # Update mean
            x_new = np.sum(w[:mu, np.newaxis] * offspring[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma*sqrt(mu_eff)*C^{-1/2}*(x_new - x_mean)/sigma
            # We need C^{-1/2}. We can use the Cholesky factor L. C^{-1/2} = L^{-T}
            try:
                L = np.linalg.cholesky(C)
                # Solve L^T z = v
                v = (x_new - x_mean) / sigma
                z = np.linalg.solve(L.T, v)
            except np.linalg.LinAlgError:
                U, s, Vt = np.linalg.svd(C)
                L_inv_T = Vt.T @ np.diag(1/np.sqrt(s)) @ U.T
                v = (x_new - x_mean) / sigma
                z = L_inv_T @ v
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z
            
            # Hessian norm
            hs = np.sqrt((p_sigma * (p_sigma + (d_sigma - 1) * p_sigma) * p_sigma) / (1 - (1 - c_sigma) ** (2 * (gen + 1))))
            sigma *= np.exp(c_sigma / chi_n * (hs - chi_n))
            
            # Update covariance
            # p_c = (1-c_c)*p_c + c_c*sqrt(mu_eff)*(x_new - x_mean)/sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (x_new - x_mean) / sigma
            
            # Rank-1 and Rank-mu update
            delta_sig = (hs > (1 + (2 / (n + 1) - 1) * (1 - c_c)) * chi_n)
            C = (1 + c_1 * (delta_sig - 1) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu part
            y_k = (offspring[:mu] - x_mean) / sigma
            C += c_mu * np.sum([w[i] * np.outer(y_k[i], y_k[i]) for i in range(mu)], axis=0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new
            gen += 1

        # If we have remaining budget and time, do a simple (1+1)-ES polish
        if remaining > 0 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(remaining, 50)): # Limit polish steps to avoid wasting time
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x = x
                    best_f = fx
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
