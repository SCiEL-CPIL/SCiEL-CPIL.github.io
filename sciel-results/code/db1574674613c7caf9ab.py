import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        total_budget = self.budget
        
        # Phase 1: CMA-ES
        # Use 80% of the budget for global search
        cma_budget = int(total_budget * 0.8)
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        # Cholesky helper
        def update_cholesky(C_new):
            try:
                L_new = np.linalg.cholesky(C_new)
                return L_new
            except np.linalg.LinAlgError:
                C_reg = C_new + 1e-4 * np.eye(dim)
                try:
                    return np.linalg.cholesky(C_reg)
                except:
                    return np.linalg.cholesky(np.eye(dim))

        L = update_cholesky(C)
        
        while calls_used + lam <= cma_budget:
            if time.time() > deadline:
                break
                
            L = update_cholesky(C)
            
            z = np.random.randn(lam, dim)
            y = z @ L.T
            x_pop = x_mean + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            f_pop = np.array([func(x) for x in x_pop])
            calls_used += lam
            
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()
                
            idx_sort = np.argsort(f_pop)
            x_sorted = x_pop[idx_sort]
            y_sorted = y[idx_sort]
            
            x_w = x_sorted[:mu]
            y_w = y_sorted[:mu]
            
            x_mean_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)
            d_mean = x_mean - x_mean_old
            
            try:
                L_inv_T = np.linalg.inv(L).T
            except:
                L_inv_T = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
            gen = calls_used / lam
            c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)
            
            if gen > 0:
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 / gen))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
            else:
                h_sigma = 1.0
                
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + c_mu * np.sum(weights[:, np.newaxis]**2 * (y_w - np.mean(y_w, axis=0))[:, np.newaxis] * (y_w - np.mean(y_w, axis=0)), axis=0)
            
            C = (C + C.T) / 2

        # Phase 2: Nelder-Mead Polish
        # Use remaining budget
        remaining_budget = total_budget - calls_used
        
        if remaining_budget > 0 and time.time() < deadline:
            # Initialize Nelder-Mead
            # Simplex: best_x and best_x + delta * e_i
            delta = 0.1 # Initial step size for local search
            
            simplex = [best_x.copy()]
            f_vals = [best_f]
            
            # Generate initial simplex points
            for i in range(dim):
                x_i = best_x.copy()
                x_i[i] += delta
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)
                f_i = func(x_i)
                f_vals.append(f_i)
                calls_used += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                
                if calls_used >= total_budget or time.time() > deadline:
                    break

            # Nelder-Mead Loop
            # We need to manage the simplex carefully
            # Standard NM:
            # 1. Sort simplex by f
            # 2. Compute centroid
            # 3. Reflect
            # 4. Expand or Contract or Shrink
            
            nm_delta = delta
            
            while calls_used < total_budget and time.time() < deadline:
                # Sort simplex
                sort_idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in sort_idx]
                f_vals = [f_vals[i] for i in sort_idx]
                
                # Check convergence: if best and worst are very close
                if abs(f_vals[0] - f_vals[-1]) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_ref = centroid - (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1
                
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < f_vals[0]:
                    # Expand
                    x_exp = centroid + 1.5 * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_used += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                        if f_exp < best_f:
                            best_f = f_exp
                            best_x = x_exp.copy()
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contract
                    # Outside contraction if f_ref < f_vals[-1] (worst), else inside
                    if f_ref < f_vals[-1]:
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        x_con = centroid - 0.5 * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1
                    
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shrink = func(simplex[i])
                            calls_used += 1
                            f_vals[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                        
                        if calls_used >= total_budget or time.time() > deadline:
                            break

        return best_x, best_f
