import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # 1. Initialization: Find a good starting point
        # Use 5 random points to initialize
        n_init = 5
        if budget < n_init + 10:
             # Fallback to simple random search if budget is tiny
             best_x = None
             best_f = np.inf
             for _ in range(budget):
                 x = np.random.uniform(-5, 5, n)
                 f = func(x)
                 if f < best_f:
                     best_f = f
                     best_x = x
             return best_x, best_f

        x_inits = np.random.uniform(-5, 5, (n_init, n))
        f_inits = np.array([func(x) for x in x_inits])
        best_idx = np.argmin(f_inits)
        m = x_inits[best_idx].copy()
        best_f = f_inits[best_idx]
        
        # 2. CMA-ES Setup
        # Remaining budget
        remaining = budget - n_init
        
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 4: lam = 4
        
        # Parameters
        mu = lam // 2
        # Logarithmic weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_w = np.sum(w * w)
        
        chi_lam = np.sqrt(lam) * (1 - 1/(2*lam) + 1/(12*lam**2))
        c_sigma = (mu_w + 2) / (lam/2 + mu_w + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1)/(lam/2 + 1) - 1) + c_sigma - 1)
        c_c = (4 - lam/4) / (lam + 4.2)
        c_m = (2 + 2 * min(0, c_c - 1) + mu_w/(lam/2)) / (lam/2 + 2 + c_c + mu_w/(lam/2))
        
        # State variables
        sigma = 1.0  # Initial step size, maybe scale with box size?
        # Box is [-5,5], size 10. Initial sigma 1.0 is reasonable.
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls_used = n_init
        best_x = m.copy()
        
        while calls_used + lam <= budget:
            # Sample lambda points
            z = np.random.randn(lam, n)
            # Transform z using C_sqrt_inv
            # Since C is symmetric positive definite, we can use Cholesky or eigendecomp
            # For simplicity and speed in low dim, we maintain C and its inverse sqrt via eigendecomp each step? 
            # Or just use C directly if we update it carefully.
            # Standard CMA-ES uses C^-1/2.
            # Let's compute eigendecomposition of C to get C^-1/2
            w_eig, v_eig = np.linalg.eigh(C)
            # Ensure positive
            w_eig = np.clip(w_eig, 1e-12, None)
            C_inv_sqrt = (v_eig / np.sqrt(w_eig)) @ v_eig.T
            
            y = z @ C_inv_sqrt.T  # Shape (lam, n)
            x_pop = m + sigma * y
            
            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_used += lam
            
            # Update best
            pop_best_idx = np.argmin(f_pop)
            if f_pop[pop_best_idx] < best_f:
                best_f = f_pop[pop_best_idx]
                best_x = x_pop[pop_best_idx].copy()
            
            # Sort population
            idx_sorted = np.argsort(f_pop)
            x_sorted = x_pop[idx_sorted]
            f_sorted = f_pop[idx_sorted]
            
            # Update mean
            m_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update covariance
            # y_diff = (x_sorted[:mu] - m) / sigma
            y_diff = (x_sorted[:mu] - m) / sigma
            C_new = (1 - c_m * (2 - c_m)) * C
            C_new += c_m * (w[:, None] * (y_diff @ y_diff.T) * w[None, :]) # Wait, formula is sum w_i (y_i - y_bar)(y_i - y_bar)^T?
            # Standard formula: C <- (1 - c_m(2-c_m))C + c_m sum w_i y_i y_i^T (where y_i are centered by old mean? No, centered by new mean? No, y_i = (x_i - m_old)/sigma)
            # Actually, the standard update is:
            # C <- (1 - c_m (2 - c_m)) C + c_m * sum_{i=1..mu} w_i (y_i - y_bar)(y_i - y_bar)^T ? 
            # No, it's usually:
            # C <- (1 - c_m (2 - c_m)) C + c_m * H_sigma
            # where H_sigma is the empirical covariance of the selected y's.
            # Let's use the standard formulation from Hansen's paper.
            # H_sigma = sum w_i (y_i - y_bar) (y_i - y_bar)^T is NOT correct.
            # It is: C <- (1 - c_m (2 - c_m)) C + c_m * (1/mu_w) * sum w_i y_i y_i^T ? No.
            
            # Let's use a simpler, robust update for C:
            # y_centered = y_diff - np.mean(y_diff * w[:, None], axis=0)
            # H = sum w_i * outer(y_i, y_i)
            # C = (1 - c_m * (2 - c_m)) * C + c_m * H
            
            y_bar = np.sum(w[:, None] * y_diff, axis=0)
            # Corrected y_diff for covariance update should be centered?
            # In CMA-ES, the update is:
            # C <- (1 - c_m (2 - c_m)) C + c_m * sum w_i (y_i - y_bar)(y_i - y_bar)^T is WRONG.
            # The term is based on the deviations from the OLD mean m.
            # So H_sigma = sum w_i y_i y_i^T where y_i = (x_i - m_old)/sigma.
            H_sigma = np.sum(w[:, None] * (y_diff * y_diff.T * w[None, :]), axis=0) # This is wrong syntax
            H_sigma = np.zeros((n, n))
            for i in range(mu):
                H_sigma += w[i] * np.outer(y_diff[i], y_diff[i])
            
            C = (1 - c_m * (2 - c_m)) * C + c_m * H_sigma
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_w) / chi_lam * (y_bar)
            # Note: y_bar is the mean of the selected y's.
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lam - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c
            if np.linalg.norm(p_sigma) / chi_lam < 1.4 + c_c:
                p_c += c_c * (np.sqrt(mu_w) / chi_lam) * (y_bar)
                
            # Update C using p_c
            C += c_c * (2 - c_c) * (np.outer(p_c, p_c) - C)
            
            m = m_new
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
        return best_x, best_f
