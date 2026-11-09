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
        # Population size: 4 + 3 * ln(n) is standard, but for small n=5, N=10 is often better for exploration/exploitation balance in limited budgets.
        # Let's use N = 10.
        N = 10
        # Initial step size: The box is [-5, 5]. A reasonable start is to cover a significant portion, e.g., sigma = 2.0.
        sigma = 2.0
        # Initial mean: Origin is a good guess, though hidden shift exists.
        x_mean = np.zeros(dim)
        # Covariance matrix: Identity
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        C_eigenvalues = np.ones(dim)
        C_eigenvectors = np.eye(dim)
        
        # CMA-ES Parameters (standard recommendations)
        # mu = N // 2
        mu = N // 2
        # Weights: log-uniform or linear? Log-uniform is standard for CMA-ES.
        # w_i = log(mu + 0.5) - log(i + 1)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # mu_eff = 1 / sum(w^2)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # c_sigma: step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # d_sigma: damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # c_c: covariance adaptation (CM)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # c_1: rank-1 update
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        # c_mu: rank-mu update
        c_mu = (mu_eff - 2 + 1 / c_1) / ((dim + 2)**2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        # Chi-square expectation
        chi_N = np.sqrt(dim * (1 - 1/(4*dim) + 1/(21*dim**2)))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Sampling helper
        def sample_from_C(sigma_val):
            # z ~ N(0, I)
            z = np.random.randn(dim)
            # y = C^{1/2} z
            # We maintain C_eigenvalues and C_eigenvectors to compute C^{1/2} efficiently
            # C = V * diag(eigenvalues) * V^T
            # C^{1/2} = V * diag(sqrt(eigenvalues)) * V^T
            # For dim=5, this is cheap.
            sqrt_eigs = np.sqrt(C_eigenvalues)
            # y = V * (sqrt_eigs * z)
            y = C_eigenvectors @ (sqrt_eigs * z)
            return sigma_val * y

        # Main Loop
        # Each generation consumes N evaluations.
        # Total generations = budget // N
        max_gen = budget // N
        
        # We will run until budget is exhausted or time runs out.
        for gen in range(max_gen):
            if time.time() > self.deadline:
                break
            if self.calls + N > self.budget:
                # If we don't have enough budget for a full generation, 
                # we can still do a partial one or stop. 
                # Stopping is safer to avoid complex partial logic, 
                # but we might waste < N calls. 
                # Let's just stop if we can't do a full gen.
                break
            
            # 1. Sample N points
            # x_i = x_mean + sigma * C^{1/2} * z_i
            samples = []
            for i in range(N):
                z = np.random.randn(dim)
                sqrt_eigs = np.sqrt(C_eigenvalues)
                y = C_eigenvectors @ (sqrt_eigs * z)
                x_i = x_mean + sigma * y
                samples.append(x_i)
            
            # 2. Evaluate
            f_values = []
            for x_i in samples:
                if time.time() > self.deadline:
                    break
                if self.calls >= self.budget:
                    break
                f_val = wrapped_func(x_i)
                f_values.append(f_val)
            
            if len(f_values) < N:
                break
                
            # 3. Sort by fitness
            # We need to reorder samples and f_values
            sorted_indices = np.argsort(f_values)
            samples = [samples[i] for i in sorted_indices]
            f_values = [f_values[i] for i in sorted_indices]
            
            # 4. Update mean
            # x_mean_new = sum(w_i * x_i) for i in 1..mu
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * samples[i]
            
            # 5. Update p_sigma and sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^{-1/2} * (x_mean_new - x_mean))
            
            # Compute C^{-1/2} (x_mean_new - x_mean)
            # C^{-1/2} = V * diag(1/sqrt(eigs)) * V^T
            diff = x_mean_new - x_mean
            inv_sqrt_eigs = 1.0 / np.sqrt(C_eigenvalues)
            # C^{-1/2} diff = V @ (inv_sqrt_eigs * (V^T @ diff))
            C_inv_sqrt_diff = C_eigenvectors @ (inv_sqrt_eigs * (C_eigenvectors.T @ diff))
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt_diff
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_N - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_N - 1))
            
            # 6. Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # h_sigma = 1 if ||p_sigma||^2 / (1 - (1-c_sigma)^{2(gen+1)}) < (1+2/c_sigma) * chi_N^2 * (1 + 0.3 * (mu_eff / dim - 1))
            # For simplicity and stability with small dim, we often approximate or just use the condition.
            # Let's use a simplified check or just assume h_sigma=1 if norm is reasonable.
            # A more robust check:
            # But for dim=5, we can compute the exact threshold if we track the generation count for the power term.
            # Since we don't track exact power history easily in this loop structure without extra state, 
            # and dim is small, let's use the standard approximation.
            # Actually, let's just set h_sigma = 1.0 for now, as it's a common simplification in short runs 
            # or we can implement the check.
            # Let's implement the check properly.
            # We need to track the age of p_sigma effectively.
            # Simpler: if norm_p_sigma is not too large, h_sigma=1.
            # Threshold: (1 + 2/c_sigma) * chi_N^2
            threshold = (1 + 2.0/c_sigma) * chi_N**2
            # The (1 - (1-c_sigma)^(2*(gen+1))) term is close to 1 for large gen, but small for small gen.
            # For gen=0, it's small. This might cause h_sigma=0 early on incorrectly?
            # No, if the numerator is small, the ratio is small, so h_sigma=1.
            # If the ratio > threshold, h_sigma=0.
            # We need the denominator: 1 - (1-c_sigma)^(2*(gen+1))
            denom = 1 - (1 - c_sigma)**(2 * (gen + 1))
            if denom > 1e-10:
                ratio = (norm_p_sigma**2) / denom
                h_sigma = 1.0 if ratio < threshold else 0.0
            else:
                h_sigma = 1.0
                
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * p_c p_c^T + c_mu * sum(w_i * y_i y_i^T)
            # Where y_i = C^{-1/2} (x_i - x_old)
            
            # Compute y_i for the top mu individuals
            sum_yyT = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_inv_sqrt_diff # Wait, C_inv_sqrt_diff was for the mean diff.
                # We need C^{-1/2} (x_i - x_mean_old)
                diff_i = samples[i] - x_mean
                y_i = C_eigenvectors @ (inv_sqrt_eigs * (C_eigenvectors.T @ diff_i))
                sum_yyT += weights[i] * np.outer(y_i, y_i)
            
            # C_new = (1 - c_1 - c_mu) * C + c_1 * outer(p_c, p_c) + c_mu * sum_yyT
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum_yyT
            
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Eigen-decompose to update C_eigenvalues and C_eigenvectors
            # This is the most expensive part, but for dim=5, it's negligible.
            C_eigenvalues, C_eigenvectors = np.linalg.eigh(C_new)
            # Clip eigenvalues to be positive
            C_eigenvalues = np.clip(C_eigenvalues, 1e-10, None)
            
            # Update state
            C = C_new
            x_mean = x_mean_new
            
            # Optional: If sigma becomes too small or too large, handle it?
            # CMA-ES usually handles this.
            
            # Check for convergence?
            # If sigma < 1e-8, we can stop.
            if sigma < 1e-8:
                break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
