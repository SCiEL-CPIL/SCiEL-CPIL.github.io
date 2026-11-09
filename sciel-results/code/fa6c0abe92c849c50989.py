"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Number of parents
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mu_eff = 1 / np.sum(weights**2)
        
        # Initial step size
        # Start with a reasonable step size for the box [-5, 5]
        sigma = 1.0
        
        # Initial mean
        # The minimum is in [-4, 4]^5. Start at 0.
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # Call counter
        calls = 0
        best_x = m.copy()
        best_f = np.inf
        
        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val
        
        # Main loop
        while calls < budget:
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(dim)
                L = np.eye(dim)
            
            ys = []
            xs = []
            fs = []
            
            # Sample lambda_ points
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.randn(dim)
                y = sigma * (L @ z)
                x = m + y
                # Clip to box for evaluation, but keep x for CMA updates?
                # Standard CMA-ES is unconstrained. Clipping x for func is fine.
                # But for the update, we should probably use the unclipped x 
                # or handle the constraint. Given the box is large enough 
                # and min is inside, clipping the *input* to func is sufficient.
                # The CMA-ES mean might drift out, so we clip m periodically.
                
                f_val = evaluate(x)
                ys.append(y)
                xs.append(x)
                fs.append(f_val)
                
            if not fs:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            ys = np.array([ys[i] for i in indices])
            xs = np.array([xs[i] for i in indices])
            fs = np.array([fs[i] for i in indices])
            
            # Update mean
            # Use top mu
            if len(ys) >= mu:
                y_parents = ys[:mu]
                m_new = m + np.sum(weights[:, np.newaxis] * y_parents, axis=0)
                
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (np.sum(weights[:, np.newaxis] * y_parents, axis=0) / sigma))
                
                # Heaviside function for sigma adaptation
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1-c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                    
                # Update p_c
                p_c = (1 - c_c) * p_c + c_c * (np.linalg.norm(p_sigma) / chi_n > 1.4 + 2/(dim+1)) * (np.sum(weights[:, np.newaxis] * y_parents, axis=0) / sigma)
                
                # Update C
                rank_mu = np.outer(np.sum(weights[:, np.newaxis] * y_parents, axis=0) / sigma, 
                                   (np.sum(weights[:, np.newaxis] * y_parents, axis=0) / sigma).T)
                rank_one = np.outer(p_c, p_c)
                C = (1 + c_1 + c_mu - c_c - c_1*c_c/2) * C + c_1 * (rank_one - C) + c_mu * rank_mu
                
                m = m_new
                
                # Keep mean within reasonable bounds
                m = np.clip(m, -4.5, 4.5)
                
                # Ensure C is symmetric and positive semi-definite
                C = (C + C.T) / 2
                # Regularize C if needed
                # Check eigenvalues
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                         C += 1e-10 * np.eye(dim)
                except:
                    pass
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
