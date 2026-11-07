import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0  # Safety margin before 300s limit
        
        # CMA-ES parameters
        # For n=5, standard lambda is often 4 + 3*log(n) ~ 9-10
        lam = int(4 + 3 * np.log(n))
        if lam < 4: 
            lam = 4
        mu = lam // 2
        
        # Weights for the top mu individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Evolution path and covariance matrix initialization
        m = np.zeros(n)
        sigma = 2.0
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        
        rng = np.random.default_rng(42)
        
        evals = 0
        best_x = None
        best_f = np.inf
        
        # Initial evaluation at center
        x_init = m.copy()
        f_init = func(x_init)
        evals += 1
        best_x = x_init.copy()
        best_f = f_init
        
        # Pre-compute Cholesky for C = I
        C_sqrt = np.linalg.cholesky(C)
        inv_C_sqrt = np.linalg.inv(C_sqrt)
        
        # Main Loop
        while evals + lam <= budget:
            if time.time() - t_start > max_time:
                break
                
            # Generate offspring
            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform to normal with covariance C
                y = C_sqrt @ z
                x = m + sigma * y
                X[i] = x
                F[i] = func(x)
                evals += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update Mean
            y_weighted = np.zeros(n)
            for i in range(mu):
                y_weighted += w[i] * (X[i] - m) / sigma
            
            # Update Step-size Control (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Check condition for sigma update
            # The denominator is the expected norm of ps if C=I, approximated
            # We use the standard condition: ||ps|| / chi_n < h_sigma * (1.4 + 2/(n+1))
            # h_sigma is 1 if ||ps||/chi_n < (1.4 + 2/(n+1)), else 0.
            # A more robust check often used in implementations:
            h_sigma = 1 if (np.linalg.norm(ps) / chi_n) < (1.4 + 2/(n+1)) else 0
            
            if h_sigma == 1:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Covariance
            pc = (1 - c_c) * pc + c_c * y_weighted
            
            # Rank-1 update
            delta_sigma = c_sigma * (2 - c_sigma) * mueff
            C_new = (1 - c_1 - c_mu) * C 
            C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)
            
            # Rank-mu update
            z = (X[:mu] - m) / sigma
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(z[i], z[i])
                
            # Ensure positive semi-definite
            C_new = (C_new + C_new.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                inv_C_sqrt = np.linalg.inv(C_sqrt)
                C = C_new
            except np.linalg.LinAlgError:
                # Fallback: keep old C or add more to diagonal
                C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(n))
                inv_C_sqrt = np.linalg.inv(C_sqrt)
                # C remains unchanged in this iteration effectively, or we could reset C to I if it's bad
                pass
            
            # Update Mean
            m = m + sigma * y_weighted
            
            # Convergence check: if sigma is very small, we are likely done
            if sigma < 1e-10:
                break

        return best_x, best_f
