"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin: The evaluator kills after 330s. We stop at 290s.
        start_time = time.time()
        deadline = start_time + 290.0

        # CMA-ES Parameters
        # For dim=5, a population size of 4 + 3*ln(5) ~ 9 is standard.
        # We use a slightly larger pop for better global exploration in the first phase.
        lambda_ = 10 
        mu = lambda_ // 2  # Weighted recombination uses the best mu individuals
        
        # Initial parameters
        # The problem states the minimizer is in [-4, 4]^5.
        # We start at the center (0,0,0,0,0) with a moderate initial sigma.
        x_mean = np.zeros(dim)
        sigma = 1.5  # Initial step size
        
        # CMA-ES adaptation rates
        c_sigma = (mu_eff + 2) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Calculate mu_eff (effective population size)
        # Weights are proportional to rank
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Recalculate rates with correct mu_eff
        c_sigma = (mu_eff + 2) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for covariance
        ps = np.zeros(dim)  # Evolution path for step size
        
        # Pre-compute sqrt(C) using Cholesky decomposition for efficiency
        # We will update C and re-decompose every generation
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0

        # Initial evaluation of the mean (counts as 1 call)
        # Note: In strict CMA-ES, we evaluate the population. 
        # To save budget and ensure we have a valid best_x early, we evaluate x_mean first.
        f_mean = func(x_mean)
        calls_made += 1
        best_f = f_mean
        best_x = x_mean.copy()

        # Main Loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # 1. Sample population
            # We need to decompose C to get the transformation matrix
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B @ B.T)) # Actually Cholesky gives L, L@L.T = C. 
                # Standard CMA-ES uses B and D such that C = B @ diag(D^2) @ B.T
                # Let's use a simpler approach: 
                # Eigen decomposition is slower but robust. 
                # For dim=5, Cholesky is fast.
                # C = L @ L.T. We sample y ~ N(0, I). x = x_mean + sigma * L @ y.
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = np.eye(dim)
                L = np.eye(dim)
            
            population = []
            fvals = []
            
            # Generate lambda_ points
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = np.random.randn(dim)
                x_i = x_mean + sigma * (L @ y)
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls_made += 1
                
                population.append(x_i)
                fvals.append(f_i)
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(population) < lambda_:
                break

            # 2. Sort population by fitness
            indices = np.argsort(fvals)
            population = [population[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # 3. Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i]
            
            # 4. Update step size sigma
            # Calculate mean of the old population (weighted)
            # Actually, standard CMA-ES uses the unweighted mean of the selected points for pc update?
            # No, it uses the weighted mean x_new.
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mu_eff) * (B.T @ (D^-1 * (x_new - x_mean) / sigma))
            
            # We need B and D from C = B @ diag(D^2) @ B.T
            # We used L = cholesky(C). So C = L @ L.T.
            # This is not the standard B, D form directly unless we orthogonalize.
            # However, for the update of ps, we need the inverse of the transformation.
            # Let's stick to the simpler "Eigen" update if Cholesky is tricky for the inverse.
            # Or, we can approximate the update.
            # Given dim=5, let's use Eigen decomposition for robustness in the update step.
            # It's slower but safer for correctness in a small dimension.
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D_eig = np.sqrt(eigvals)
            
            # Transform x_new - x_mean into the eigenbasis
            # v = (x_new - x_mean) / sigma
            v = (x_new - x_mean) / sigma
            # Project onto eigenvectors
            v_eig = eigvecs.T @ v
            # Scale by 1/D
            v_eig /= D_eig
            # Project back
            v_scaled = eigvecs @ v_eig
            
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * v_scaled
            ps_norm = np.linalg.norm(ps)
            
            # hsig is the Heaviside function indicating if ps is significantly different from 0
            hsig = (ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lambda_)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
            
            # 5. Update evolution path for covariance pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_scaled
            
            # 6. Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i - x_new) @ (y_i - x_new).T)
            
            # Rank-1 update
            C = (1 - c_1 - c_mu) * C
            if hsig:
                C += c_1 * (pc @ pc.T)
            else:
                C += c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            for i in range(mu):
                diff = (population[i] - x_new) / sigma
                # Transform diff to eigenbasis? No, the formula uses the raw difference in the current frame.
                # Actually, the standard formula is:
                # C += c_mu * sum( w_i * (z_i - z_mean) @ (z_i - z_mean).T ) where z are the standardized variables.
                # z_i = (x_i - x_mean) / sigma.
                # We need the outer product of the standardized differences.
                z_i = (population[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(z_i, z_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # 7. Update mean
            x_mean = x_new
            
            # 8. Adaptation of sigma
            # sigma = sigma * exp(c_sigma / (5 * dim) * (ps_norm / sqrt(mu_eff) - 1))
            sigma = sigma * np.exp(c_sigma / (5 * dim) * (ps_norm / np.sqrt(mu_eff) - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check for stagnation or convergence
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
