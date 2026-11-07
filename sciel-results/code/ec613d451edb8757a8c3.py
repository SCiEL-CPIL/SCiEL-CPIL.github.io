import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_ = 10
        # Number of selected individuals
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance effective selection mass
        mueff = 1 / np.sum(weights**2)
        
        # Initial mean and sigma
        rng = np.random.default_rng(0)
        m = np.zeros(dim)
        # Start with a sigma that covers the box reasonably
        # Box is [-5, 5], width 10. Sigma 1.0 is 10% of width.
        # Maybe 2.0 is better to explore quickly?
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Time constants
        # Standard formulas
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (dim + 2) - 1))
        
        # State
        best_x = m.copy()
        best_f = func(best_x)
        evals = 1
        
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Sample
            z = rng.standard_normal((lambda_, dim))
            # Transform
            # x = m + sigma * C_sqrt @ z.T
            x = (sigma * (C_sqrt @ z.T).T).T + m
            # Clip to box
            x = np.clip(x, -5, 5)
            
            # Evaluate
            fvals = np.array([func(xi) for xi in x])
            evals += lambda_
            
            # Sort
            order = np.argsort(fvals)
            x_sorted = x[order]
            f_sorted = fvals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma
            inv_sigma = 1.0 / sigma if sigma > 1e-12 else 0.0
            # C_sqrt @ z is the direction in the transformed space
            # Actually, the standard update is:
            # p_sigma = (1-c_sigma)p_sigma + c_sigma * sqrt(mueff) * inv_sigma * C_sqrt^-1 * (m - m_old)
            # But C_sqrt^-1 is expensive. We can use the fact that z is standard normal.
            # The direction in z-space is (m - m_old) / sigma * C_sqrt^-1.
            # However, a common approximation or alternative is to use the weighted sum of z's.
            # Let's stick to the standard formula.
            # We need C_sqrt_inv.
            # Since C is updated, C_sqrt changes.
            # For efficiency in 5D, we can just compute the inverse of C_sqrt or use np.linalg.solve.
            # But wait, C_sqrt is from the *previous* iteration.
            
            # Let's compute the direction in the standard normal space.
            # d = (m - m_old) / sigma
            # We need C_sqrt^-1 @ d
            try:
                d_z = np.linalg.solve(C_sqrt, (m - m_old) / sigma)
            except np.linalg.LinAlgError:
                d_z = (m - m_old) / sigma
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * d_z
            
            # Hessian norm check for sigma update
            hs = np.sqrt(np.sum(p_sigma**2) / dim)
            # c_hsig = 1 + 2 * max(0, sqrt((mueff-1)/(dim+1))-1) + c_sigma
            # E[||z||] approx
            E_norm_z = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            if hs < (1.4 + 2/(dim+1)) * E_norm_z:
                sigma *= np.exp(c_sigma/d_sigma * (hs/E_norm_z - 1))
            
            # p_c
            # p_c = (1-c_c)p_c + h_sig * c_c * sqrt(mueff) * inv_sigma * C_sqrt^-1 * (m - m_old)
            # h_sig = 1 if hs < (1.4 + 2/(dim+1)) * E_norm_z else 0
            h_sig = 1.0 if hs < (1.4 + 2/(dim+1)) * E_norm_z else 0.0
            p_c = (1 - c_c) * p_c + h_sig * c_c * np.sqrt(mueff) * d_z
            
            # Update Covariance
            # Delta H = c_1 * (p_c p_c^T + (1 - h_sig) * c_c * (2 + c_c) * C - C) 
            #            + c_mu * sum w_i (y_i y_i^T)
            # where y_i = (x_i - m_old) / sigma
            
            y = (x_sorted[:mu] - m_old) / sigma
            # Weighted sum of outer products
            delta_H = np.zeros((dim, dim))
            for i in range(mu):
                delta_H += weights[i] * np.outer(y[i], y[i])
            
            # Rank-one update part
            rank_one = np.outer(p_c, p_c)
            if h_sig < 1.0:
                rank_one += (1 - h_sig) * c_c * (2 + c_c) * (C - np.eye(dim)) # Wait, formula is (1-h_sig)c_c(2+c_c)C - (1-h_sig)c_c(2+c_c)I?
                # Standard formula: (1-h_sig) c_c (2+c_c) C - (1-h_sig) c_c (2+c_c) I? No.
                # It is: (1-h_sig) * c_c * (2 + c_c) * C - (1-h_sig) * c_c * (2 + c_c) * I
                # Actually, the term is: (1-h_sig) * c_c * (2 + c_c) * (C - I) is NOT correct.
                # The term is: (1-h_sig) * c_c * (2 + c_c) * C - (1-h_sig) * c_c * (2 + c_c) * I
                # Let's check the standard CMA-ES paper.
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sig) c_c (2+c_c) C) + c_mu sum w_i y_i y_i^T
                # This expands to:
                # C_new = (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_1 (1-h_sig) c_c (2+c_c) C + c_mu sum w_i y_i y_i^T
                # Coeff of C: 1 - c_1 - c_mu + c_1 (1-h_sig) c_c (2+c_c)
                
                # Let's just use the direct update formula from the paper:
                # C += -c_1*C + c_1*(p_c p_c^T + (1-h_sig)*c_c*(2+c_c)*C) + c_mu*sum(w_i y_i y_i^T)
                # This is equivalent to:
                # C = (1 - c_1 + c_1*(1-h_sig)*c_c*(2+c_c) - c_mu) * C + c_1*p_c p_c^T + c_mu*sum(...)
                
                pass # Handled below
            
            # Correct update:
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 + c_c) * C)
            C += c_mu * delta_H
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update C_sqrt
            # Use Cholesky or Eigen
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, use eigen
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                # Cholesky requires lower triangular, Eigen gives symmetric.
                # We need a matrix M such that M @ M.T = C.
                # If we use Eigen, M = eigvecs @ sqrt(eigvals).
                # Then M @ M.T = eigvecs @ sqrt(eigvals) @ sqrt(eigvals) @ eigvecs.T = eigvecs @ eigvals @ eigvecs.T = C.
                # So C_sqrt should be eigvecs @ np.diag(np.sqrt(eigvals)).
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

        return best_x, best_f
