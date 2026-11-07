"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with multi-start and local polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill

        # Weierstrass f16 has multi-scale features.
        # Use a population-based search with multiple scales.
        
        # Strategy: 
        # 1. Global exploration with CMA-ES using a moderate initial sigma.
        # 2. Multiple restarts if budget allows, or a single long run.
        # 3. Local polish with Nelder-Mead or similar.
        
        # Since budget is 500 and dim is 5, we can do ~50 generations of lambda=10.
        # Or ~60 generations of lambda=8.
        
        # Let's use a robust CMA-ES implementation.
        
        def cmaes(func, x0, sigma0, budget, dim, rng, time_limit, start_time):
            # Parameters
            lam = 4 + int(3 * np.log(dim))  # 8 for dim=5
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
            
            best_x = m.copy()
            best_f = func(best_x)
            evals_used = 1
            
            while evals_used + lam <= budget:
                if time.time() - start_time > time_limit:
                    break
                    
                ys = rng.standard_normal((lam, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                zs = ys @ L.T
                xs = m[None, :] + sigma * zs
                xs = np.clip(xs, -5, 5)
                
                fs = np.array([func(x) for x in xs])
                evals_used += lam
                
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = xs[idx_best].copy()
                    
                idx_sorted = np.argsort(fs)
                y_sorted = ys[idx_sorted]
                x_sorted = xs[idx_sorted]
                
                new_m = np.dot(weights, x_sorted[:mu])
                
                diff = (new_m - m) / sigma
                v = np.linalg.solve(L.T, diff)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
                
                hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
                
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
                sigma = max(sigma, 1e-10)
                
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
                
                rank1_update = np.outer(pc, pc)
                v_mu = (x_sorted[:mu] - m) / sigma
                rankmu_update = np.dot((weights[:, None] * v_mu).T, v_mu)
                
                C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
                C = (C + C.T) / 2.0
                
                m = new_m
            
            return best_x, best_f, evals_used

        def local_polish(func, x0, budget, dim, rng, time_limit, start_time):
            # Simple local search with adaptive step size
            x = x0.copy()
            f = func(x)
            step = 0.1
            evals_used = 1
            no_improve = 0
            
            while evals_used < budget:
                if time.time() - start_time > time_limit:
                    break
                
                # Try small perturbations
                improved = False
                for i in range(dim):
                    for sign in [-1, 1]:
                        x_new = x.copy()
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5, 5)
                        f_new = func(x_new)
                        evals_used += 1
                        if f_new < f:
                            x = x_new
                            f = f_new
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    step *= 1.2
                    no_improve = 0
                else:
                    step *= 0.7
                    no_improve += 1
                    if step < 1e-8:
                        break
                    
            return x, f, evals_used

        # Multi-start strategy
        # Use different initial sigmas to handle multi-scale features
        rng = np.random.default_rng(42)
        
        # Split budget: 60% global, 40% local
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget
        
        # Try multiple starting points with different scales
        # Weierstrass f16 has features at scales 1, 1/2, 1/4, ...
        # Start with sigma = 1.0, then 0.5, then 0.25
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        # First run: global exploration
        x0 = np.zeros(dim)
        sigma0 = 1.0
        bx, bf, be = cmaes(func, x0, sigma0, global_budget, dim, rng, time_limit, start_time)
        best_x = bx
        best_f = bf
        total_evals = be
        
        # Local polish
        if time.time() - start_time < time_limit - 5:
            lx, lf, le = local_polish(func, best_x, local_budget, dim, rng, time_limit, start_time)
            if lf < best_f:
                best_x = lx
                best_f = lf
            total_evals += le
        
        # If we have time and budget left, try another scale
        if total_evals < budget and time.time() - start_time < time_limit - 10:
            remaining = budget - total_evals
            if remaining > 50:
                # Try a finer scale around the best point found
                x0 = best_x.copy()
                sigma0 = 0.1
                bx2, bf2, be2 = cmaes(func, x0, sigma0, remaining, dim, rng, time_limit, start_time)
                if bf2 < best_f:
                    best_x = bx2
                    best_f = bf2
                total_evals += be2
                
                # Final polish
                if time.time() - start_time < time_limit - 5:
                    remaining2 = budget - total_evals
                    if remaining2 > 10:
                        lx2, lf2, le2 = local_polish(func, best_x, remaining2, dim, rng, time_limit, start_time)
                        if lf2 < best_f:
                            best_x = lx2
                            best_f = lf2
                        total_evals += le2
        
        return best_x, best_f

# EVOLVE-BLOCK-END
