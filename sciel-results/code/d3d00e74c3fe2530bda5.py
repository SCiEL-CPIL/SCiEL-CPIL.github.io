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
        
        # Phase 1: CMA-ES for global exploration
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
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        evals_used = 0
        
        f_mean = func(x_mean)
        evals_used += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        L = np.linalg.cholesky(C)
        
        # Run CMA-ES until stagnation or 80% budget
        cma_budget_limit = int(budget * 0.8)
        stagnation_count = 0
        stagnation_threshold = 30 # No improvement in 30 evals -> switch
        
        while evals_used < cma_budget_limit:
            if time.time() > deadline:
                break
                
            if budget - evals_used < lam:
                break

            ys = np.zeros((lam, dim))
            xs = np.zeros((lam, dim))
            fs = np.zeros(lam)
            
            # Vectorized generation
            zs = rng.standard_normal((lam, dim))
            ys = zs @ L.T
            xs = x_mean + sigma * ys
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            for i in range(lam):
                if evals_used >= budget or time.time() > deadline:
                    break
                f_val = func(xs[i])
                evals_used += 1
                fs[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = xs[i].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
            if stagnation_count >= stagnation_threshold:
                break

            idx = np.argsort(fs)
            xs = xs[idx]
            ys = ys[idx]
            fs = fs[idx]
            
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update p_sigma
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ (x_mean - x_mean_old) / sigma)
            
            y_bar = (x_mean - x_mean_old) / sigma
            
            # Vectorized sum_w_y
            sum_w_y = np.sum(w[:mu, None, None] * ys[:mu, :, None] * ys[:mu, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
            
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)

        # Phase 2: Local refinement with Nelder-Mead
        remaining = budget - evals_used
        if remaining > 5 and time.time() < deadline:
            # Initialize simplex around best_x
            x0 = best_x.copy()
            f0 = best_f
            alpha = 0.1 # Initial simplex spread
            simplex = [x0]
            f_values = [f0]
            
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                evals_used += 1
                simplex.append(x_i)
                f_values.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            # Nelder-Mead iterations
            # Standard parameters
            alpha_nm = 1.0  # Reflection
            gamma_nm = 2.0  # Expansion
            rho_nm = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink
            
            while evals_used < budget and time.time() < deadline:
                # Sort simplex
                sort_idx = np.argsort(f_values)
                simplex = [simplex[i] for i in sort_idx]
                f_values = [f_values[i] for i in sort_idx]
                
                # Stop if simplex is small enough
                if np.max(f_values) - np.min(f_values) < 1e-10:
                    break
                    
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha_nm * (x_centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                evals_used += 1
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = x_centroid + gamma_nm * (x_centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    evals_used += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                elif f_r < f_values[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                else:
                    # Contraction
                    x_c = x_centroid + rho_nm * (x_worst - x_centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    evals_used += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = func(simplex[i])
                            evals_used += 1
                            if f_values[i] < best_f:
                                best_f = f_values[i]
                                best_x = simplex[i].copy()
                        
                        if evals_used >= budget:
                            break

        return best_x, best_f
