import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 9
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State
        x = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        
        best_x = x.copy()
        best_f = func(best_x)
        evals = 1
        gen = 0
        
        stagnation_count = 0
        stagnation_limit = 15 # Generations without improvement
        
        while evals < budget:
            if time.time() > deadline:
                break
                
            gen += 1
            
            # Generate Population
            ys = []
            xs = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset C
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                
            for i in range(lam):
                z = np.random.randn(dim)
                y = L @ z
                x_i = x + sigma * y
                x_i = np.clip(x_i, -5, 5)
                ys.append(y)
                xs.append(x_i)
                
            # Evaluate
            fs = []
            for x_i in xs:
                if time.time() > deadline:
                    fs.append(np.inf)
                    continue
                f_i = func(x_i)
                fs.append(f_i)
                evals += 1
                
            while len(fs) < lam:
                fs.append(np.inf)
                
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update Best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Update Mean
            y_w = sum(w[i] * ys[i] for i in range(mu))
            x = x + sigma * y_w
            
            # Update Covariance
            y_diff = [ys[i] - y_w for i in range(mu)]
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / sigma
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            if denom < 1e-12: 
                denom = 1e-12
            h_sig = 1.0 if (norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * chi_n) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            C = (1 - c_1 - c_mu + c_1 * h_sig) * C
            C += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(y_diff[i], y_diff[i])
                
            C = (C + C.T) / 2
            
            # Ensure PD
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Restart
            if stagnation_count > stagnation_limit or sigma < 1e-6:
                # Random restart
                x = np.random.uniform(-5, 5, dim)
                sigma = 1.5
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                stagnation_count = 0
                # Do not reset gen, to keep h_sig normalization stable
                
        return best_x, best_f
