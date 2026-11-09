import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        
        # Fixed anisotropic ES with large population
        # Step ellipsoidal f7 has worst-case conditioning where one axis is much steeper
        # Use a fixed covariance that emphasizes exploration along all axes but with
        # anisotropy to handle the step discontinuity
        
        # Initialize covariance to capture worst-case ellipsoidal conditioning
        # The step ellipsoidal can have eigenvalues ranging from 1 to 10^4 in worst case
        # Use a moderate anisotropy to balance exploration
        C = np.eye(dim)
        # Make first axis more exploratory (common in BBOB instances)
        C[0, 0] = 4.0
        C[1, 1] = 2.0
        
        L = np.linalg.cholesky(C)
        
        # Large population for parallel exploration
        lam = 50
        mu = lam // 2
        
        # Weights for selection
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        
        # Initial point
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        evals_used = 1
        
        # Simple (mu+lambda)-ES with fixed covariance
        # No adaptive updates - just selection and recombination
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_val = func(x_new)
                evals_used += 1
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_new.copy()
                    
            if len(xs) < lam:
                break
                
            xs_arr = np.array(xs)
            fs_arr = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs_arr)
            xs_sel = xs_arr[idx[:mu]]
            
            # Weighted recombination
            x_mean = np.sum(w[:mu, None] * xs_sel, axis=0)
            
            # Adaptive step size: reduce if progress, increase if stalled
            # Simple rule: if best improved significantly, keep sigma; else reduce
            if best_f < np.min(fs_arr) + 1e-10:
                # Had improvement in this generation
                pass
            else:
                sigma *= 0.95
            
            sigma = max(sigma, 1e-6)
            
            # Early stopping if sigma is very small
            if sigma < 1e-5:
                break
        
        # Local refinement with (1+1)-ES
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = max(sigma, 0.05)
            
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                y = L @ z
                x_new = x_local + sigma_local * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                evals_used += 1
                
                if f_new < f_local:
                    f_local = f_new
                    x_local = x_new.copy()
                    best_f = f_local
                    best_x = x_local.copy()
                    sigma_local *= 1.05
                else:
                    sigma_local *= 0.95
                    
                sigma_local = max(sigma_local, 1e-8)

        return best_x, best_f
