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
        deadline = start_time + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization
        # Sample 10 points to identify the best basin.
        # Step-ellipsoidal functions often have a "step" where the landscape changes.
        # Finding the right side of the step is crucial.
        init_candidates = []
        for _ in range(10):
            if evals_used >= budget: break
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x))
        
        init_candidates.sort(key=lambda k: k[0])
        x_mean = init_candidates[0][1].copy()
        best_f = init_candidates[0][0]
        
        # Phase 2: CMA-ES
        # Parameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Start with a moderate sigma. 1.5 is a good balance for [-5,5] box.
        sigma = 1.5 
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute L
        L = np.eye(dim)
        
        cma_budget = budget - evals_used
        cma_evals = 0
        stagnation_counter = 0
        last_best_f = best_f
        
        while cma_evals < cma_budget and evals_used < budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
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
                f_val = eval_point(x_new)
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_val)
                
            if len(xs) < lam:
                break
                
            cma_evals += lam
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            # Check for stagnation
            if fs[0] < last_best_f:
                stagnation_counter = 0
                last_best_f = fs[0]
            else:
                stagnation_counter += 1
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update C for next step's sampling
            # Calculate C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            # Update sigma
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update covariance
            sum_w_y = np.zeros((dim, dim))
            for i in range(mu):
                sum_w_y += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            # Update L
            L = np.linalg.cholesky(C)
            
            # Stop if converged or stagnated
            if sigma < 1e-4:
                break
            if stagnation_counter > 10:
                break

        # Phase 3: Local Refinement (Remaining budget)
        # Use coordinate-wise line search / Nelder-Mead hybrid
        remaining = budget - evals_used
        if remaining > 0 and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            
            # Estimate local step size
            h = max(sigma * 0.5, 0.05)
            
            # Simple coordinate descent with adaptive step size
            # This is robust for step functions where gradient might be zero or undefined
            for _ in range(remaining):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(dim):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_try = x_local.copy()
                    x_try[i] += h
                    f_try = eval_point(x_try)
                    
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        improved = True
                        # Increase step slightly if improving
                        h *= 1.1 
                    else:
                        # Try negative step
                        x_try = x_local.copy()
                        x_try[i] -= h
                        f_try = eval_point(x_try)
                        
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            improved = True
                            h *= 1.1 
                        else:
                            # Shrink step
                            h *= 0.7
                            
                    if h < 1e-5:
                        h = 0.05 # Reset if too small
                            
                if not improved and h < 0.01:
                    # Try a small random perturbation to escape local minima
                    x_try = x_local + rng.normal(0, 0.1, dim)
                    f_try = eval_point(x_try)
                    if f_try < f_local:
                        x_local = x_try
                        f_local = f_try
                        h = 0.1
                    else:
                        break # Stagnant
            
            # Update best
            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
