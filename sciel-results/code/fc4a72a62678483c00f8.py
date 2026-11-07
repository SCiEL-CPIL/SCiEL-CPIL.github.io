"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: N = 4 + 3 * ln(n)
        N = 4 + int(3 * np.log(n)) if n > 1 else 2
        # Number of best individuals to select
        mu = N // 2
        
        # Weights for the selected individuals
        weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (n + 2) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Damping factor
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initial State
        # Start at center of the box
        x = np.zeros(n)
        sigma = 2.5  # Initial step size, covers ~50% of the box width per dim
        
        # Covariance matrix and its Cholesky decomposition
        C = np.eye(n)
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(C))
        
        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Track best
        best_x = x.copy()
        best_fx = float('inf')
        
        calls = 0
        
        # Evaluate initial point
        x_init = np.clip(x, -5, 5)
        best_fx = func(x_init)
        best_x = x_init.copy()
        calls += 1
        
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = np.zeros((N, n))
            fitness = np.zeros(N)
            
            for i in range(N):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Sample from multivariate normal
                z = np.random.standard_normal(n)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                
                offspring[i] = y
                fitness[i] = func(y)
                calls += 1
                
                if fitness[i] < best_fx:
                    best_fx = fitness[i]
                    best_x = y.copy()
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort offspring by fitness
            order = np.argsort(fitness)
            y_sorted = offspring[order]
            f_sorted = fitness[order]
            
            # Update mean vector
            x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Update step size (Cumulative Step Size Adaptation)
            # z_best is the random vector that generated the best individual
            # We need to recover z from y: y = x + sigma * B @ z => z = B^-1 (y-x)/sigma
            y_best = y_sorted[0]
            step = (y_best - x) / sigma
            # Solve B z = step for z
            # Since B is lower triangular from Cholesky, we can use scipy.linalg.solve_triangular
            # But to avoid extra imports if not strictly necessary or for speed in small n, 
            # we can just do a simple back substitution or use np.linalg.solve (slower but n=5 is tiny)
            # Actually, B @ z = step => z = np.linalg.solve(B, step)
            z_best = np.linalg.solve(B, step)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_best
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (n + 1)) * chi_n
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // N))) < threshold else 0
            
            sigma = sigma * np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)
            
            # Update covariance matrix
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_best
            
            # h_c condition
            norm_p_c = np.linalg.norm(p_c)
            threshold_c = (1.4 + 2 / (n + 1)) * chi_n
            h_c = 1 if norm_p_c / np.sqrt(1 - (1 - c_c)**(2 * (calls // N))) < threshold_c else 0
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) + c_mu * sum(w_i (y_i - x)(y_i - x)^T)
            
            # Pre-compute outer products for selected individuals
            # Note: The standard update uses (y_i - x) vectors, but since y_i = x + sigma B z_i,
            # (y_i - x) = sigma B z_i. So (y_i - x)(y_i - x)^T = sigma^2 B z_i z_i^T B^T.
            # However, the standard formula often just uses the difference vectors directly.
            # Let's stick to the difference vectors for clarity and correctness with respect to the box constraints 
            # (though CMA-ES assumes unconstrained, clipping is a heuristic).
            
            # Vectorized update of C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C)
            
            # Add weighted outer products of the best mu individuals
            # We need the z vectors for the best mu individuals to update C correctly in the transformed space?
            # No, the standard CMA-ES update for C uses the *differences* in the original space 
            # if we are not doing the full "transformed" update which requires storing z.
            # Wait, the standard CMA-ES update is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + ...) + c_mu * sigma^2 * sum(w_i z_i z_i^T)
            # This is the update in the *z-space* which is then mapped back?
            # Actually, the standard implementation updates C using the *differences* y_i - x.
            # Let's check the reference. 
            # Hansen's CMA-ES:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) C) + c_mu * sum_{i=1}^mu w_i (y_{i:lambda} - m_{old})(y_{i:lambda} - m_{old})^T
            # Yes, it uses the differences in the original space.
            
            for i in range(mu):
                diff = y_sorted[i] - x
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Update x
            x = x_new
            
            # Update Cholesky decomposition
            # C might not be positive definite due to numerical errors or aggressive updates
            # We need to ensure it is PD before Cholesky
            # If Cholesky fails, we can add a small epsilon to the diagonal or reset C to identity
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, add small jitter to diagonal
                C += 1e-6 * np.eye(n)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If still failing, reset to identity (safe fallback)
                    C = np.eye(n)
                    B = np.eye(n)
                    sigma = max(sigma, 1.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
