"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    The Bent Cigar function (BBOB f12) is highly ill-conditioned.
    CMA-ES is a state-of-the-art derivative-free optimizer that adapts 
    the covariance matrix of a multivariate normal distribution to the 
    shape of the objective function. This allows it to align the search 
    directions with the eigenvectors of the Hessian (or the 'cigar' axes), 
    making it extremely efficient on ill-conditioned landscapes compared 
    to standard DE or simple local searches.
    
    Key features:
    - Adapts step size (sigma) and covariance matrix (C) based on success.
    - Small population (N=10) to maximize generations within the 500 budget.
    - Robust to the hidden shift and rotation as it learns the geometry.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        # For dim=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8. 
        # Let's use lambda=10 for better stability and exploration in early phase.
        lambda_ = 10
        mu = lambda_ // 2  # Number of best individuals for recombination
        
        # Weights for the mean update
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))

        # Initial mean: center of the box
        x_mean = np.zeros(dim)
        
        # Initial sigma: 2.0 is a reasonable start for [-5,5] box
        # The function is defined on [-5,5]^5. A start of 2.0 covers a good range.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky of C (initially I)
        C_sqrt = np.eye(dim)
        
        # Indices for the best mu individuals
        # We will sort the population by fitness
        
        generation = 0
        max_generations = budget // lambda_ + 1
        
        while self.calls < budget and time.time() < self.deadline:
            generation += 1
            if time.time() > self.deadline:
                break
            
            # 1. Sample lambda_ individuals
            # Generate random normal vectors
            N = np.random.randn(lambda_, dim)
            
            # Transform to standard normal using Cholesky
            # y_i = C_sqrt @ z_i
            # We want x_i = x_mean + sigma * C_sqrt @ z_i
            # Since C is symmetric positive definite, we can use C_sqrt
            # For efficiency, we can do (C_sqrt @ N.T).T
            
            # Note: If C becomes ill-conditioned or non-PSD due to numerical issues,
            # Cholesky might fail. We add a small jitter if needed.
            try:
                C_sqrt, _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt, _ = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if it's really broken
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
            
            # Sample points
            # N is (lambda, dim). We need C_sqrt @ N.T -> (dim, lambda) -> .T -> (lambda, dim)
            Y = (C_sqrt @ N.T).T
            X = x_mean[None, :] + sigma * Y
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                F[i] = wrapped_func(X[i])
            
            # If we broke out of the loop early due to budget/time, handle it
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            # 2. Sort individuals by fitness
            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            # 3. Update mean
            # x_mean_new = sum(weights[i] * X_sorted[i]) for i in range(mu)
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * X_sorted[i]
            
            # 4. Update sigma (CSA)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt^-1 * (x_mean_new - x_mean))
            
            # Calculate inverse of C_sqrt
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.eye(dim)
            
            # Direction in the transformed space
            delta_mean = x_mean_new - x_mean
            # h_sigma is the heuristic for the expected norm
            # chi_n is the expected norm of a standard normal vector
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Norm of the transformed mean difference
            # || C_sqrt^-1 (x_mean_new - x_mean) ||
            norm_delta = np.linalg.norm(C_sqrt_inv @ delta_mean)
            
            # h_sigma
            if norm_delta > 0:
                h_sigma = 1.0 if (norm_delta / chi_n) < (1 + 2/(dim+1)) * (1 - c_sigma) else 0.0
            else:
                h_sigma = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * delta_mean
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # 5. Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (C^-1 * (x_mean_new - x_mean))
            # Note: The update for p_c uses C^-1, not C_sqrt^-1
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ delta_mean)
            
            # Rank-1 and Rank-mu updates
            # C = (1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * (1 - c_c) * C) + c_mu * sum(weights[i] * (y_i @ y_i))
            
            # y_i are the standardized successful steps: (X_sorted[i] - x_mean) / sigma
            # Note: We use the old x_mean for standardization in the rank-mu update
            
            rank_mu_update = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (X_sorted[i] - x_mean) / sigma
                rank_mu_update += weights[i] * np.outer(y_i, y_i)
            
            # Calculate the scalar factor
            factor = 1 + c_1 * h_sigma * (2 - c_c) - c_1 - c_mu
            
            C = factor * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 - c_c) * C) + c_mu * rank_mu_update
            
            # Symmetrize to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for convergence
            # If sigma is very small or the spread of the best points is very small
            if sigma < 1e-12:
                break
            if np.max(F_sorted[:mu]) - np.min(F_sorted[:mu]) < 1e-10:
                # If we are stuck, we might want to restart, but for now, let it run.
                # CMA-ES is generally robust.
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
