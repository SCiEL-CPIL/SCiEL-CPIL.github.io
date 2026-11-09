import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # CMA-ES implementation
        # Parameters tuned for dim=5, budget=500
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # Number of best points for recombination
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance reduction constant
        mueff = 1.0 / sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial step size
        sigma = 2.0  # Start wide enough to cover [-5, 5]
        
        # Initial mean (center of search)
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Inverse square root of C for sampling
        # We update this periodically or use Cholesky decomposition each step
        # For dim=5, Cholesky is cheap.
        
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Pre-compute some constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Helper to update inverse sqrt of C
        # We can just use np.linalg.cholesky each iteration for dim=5
        
        while eval_count < budget:
            # Sample lambda points
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if eval_count >= budget:
                    break
                # Sample from N(0, C)
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize if not positive definite
                    C = C + 1e-12 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                z = rng.standard_normal(dim)
                y = L @ z
                x = mean + sigma * y
                f = func(x)
                eval_count += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(ys) == 0:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            new_mean = sum(weights[i] * xs[i] for i in range(mu))
            
            # Update evolution paths
            old_mean = mean
            mean = new_mean
            
            # p_sigma update
            norm_old_mean = np.linalg.norm(old_mean)
            # Note: The standard update uses the normalized step
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma/sigma * (new_mean - old_mean)
            # Actually, the standard formula is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff)/sigma * (new_mean - old_mean)
            # Wait, the standard CMA-ES update for p_sigma is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff) * (new_mean - old_mean) / sigma
            # Let's stick to the standard formulation from Hansen's paper.
            
            # Correction: The update for p_sigma uses the step size sigma.
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff) * (new_mean - old_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - old_mean) / sigma
            
            # p_c update
            # p_c = (1-c_c)*p_c + h_sigma * sqrt(c_c*(2-c_c)*mueff) * (new_mean - old_mean) / sigma
            # h_sigma is 1 if ||p_sigma|| / sqrt(1-(1-c_sigma)^(2*iter)) < (1.4 + 2/(dim+1)) * chi_n
            # We approximate the denominator as 1 for simplicity or track it.
            # For simplicity and robustness, we often just check ||p_sigma|| < (1.4 + 2/(dim+1)) * chi_n * (1 + (1-c_sigma)*something)
            # Let's use the standard threshold:
            # h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1-c_sigma)**(2*eval_count/lam)) < (1.4 + 2/(dim+1)) * chi_n
            # Since we don't track the exact power, we approximate the denominator by 1.
            # This is a common simplification in short-budget runs.
            threshold = (1.4 + 2.0 / (dim + 1)) * chi_n
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold * np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / lam)) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - old_mean) / sigma
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1-h_sigma)*c_c*(2-c_c)*C) + c_mu * sum(w_i * (y_i - y_w)(y_i - y_w)^T)
            
            # y_w is the weighted mean of the selected y's
            y_w = sum(weights[i] * ys[i] for i in range(mu))
            
            # Rank-1 update part
            rank1 = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu update part
            # Sum of outer products of (y_i - y_w)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys[i] - y_w
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / chi_n * (||p_sigma|| / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # Clip sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Early stopping if sigma is very small or no improvement
            if sigma < 1e-12:
                break

        return best_x, best_f
