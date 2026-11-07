import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)
        
        n = self.dim
        budget = self.budget
        
        # Track global best
        best_x = None
        best_f = float('inf')
        call_count = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, call_count
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        def time_ok():
            return time.time() < deadline
        
        # Fast CMA-ES using Cholesky for C^{1/2}
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal best_x, best_f, call_count
            
            if not time_ok() or call_count >= budget:
                return
            
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Cholesky factor of C (initially I)
            L = np.eye(n)
            
            # Evaluate initial point
            fx = func(x)
            call_count += 1
            update_best(x, fx)
            
            gen = 0
            last_improvement_gen = 0
            best_f_at_last_improve = best_f
            
            while call_count < budget and time_ok():
                # Generate population using Cholesky
                B = rng.standard_normal((lam, n))
                Y = x + sigma * (B @ L.T)
                
                # Evaluate population
                F = []
                Y_eval = []
                for i in range(lam):
                    if call_count >= budget or not time_ok():
                        break
                    y = np.clip(Y[i], -5.0, 5.0)
                    fy = func(y)
                    F.append(fy)
                    Y_eval.append(y)
                    call_count += 1
                    update_best(y, fy)
                
                if len(F) < mu:
                    break
                
                F_arr = np.array(F)
                Y_arr = np.array(Y_eval)
                
                # Sort by fitness
                idx = np.argsort(F_arr)
                Y_sorted = Y_arr[idx]
                F_sorted = F_arr[idx]
                
                # Update mean
                x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update step size
                y_invC = (Y_sorted[:mu] - x) / sigma
                # Need C^{-1/2} for ps update
                # Use Cholesky: C = L L^T, so C^{-1/2} = (L^T)^{-1} L^{-1}
                # For ps: inv_sqrt_C @ (w @ y_invC)
                # We can compute this via solving triangular systems
                # C^{-1/2} v = L^{-1} (L^T)^{-1} v
                v = w[:mu] @ y_invC  # shape (n,)
                # Solve L^T u = v
                u = np.linalg.solve(L.T, v)
                # Solve L w_ = u
                w_ = np.linalg.solve(L, u)
                # C^{-1/2} v = w_
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * w_
                
                # Update correlation matrix
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                # Update covariance
                # h_pc check
                norm_ps = np.linalg.norm(ps)
                chi_n = np.sqrt(n) * (1 + (2.0 / (n + 1)) + (1.0 / (n + 1)) * np.sqrt(2.0 * (n + 1) / (n - 1)))
                # Approximate expected norm
                expected_norm = np.sqrt(n * (1 - (1 - c_sigma)**(2 * (gen + 1))))
                threshold = (1.4 + 2.0 / (n + 1)) * np.trace(C) / n
                if norm_ps < threshold:
                    h_pc = 1.0 - np.exp(-2.0 + 5.0 / (n + 1))
                else:
                    h_pc = 0.0
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    y_i = y_invC[i]
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                C = (C + C.T) / 2
                # Add small regularization for numerical stability
                C += 1e-12 * np.eye(n)
                
                # Update Cholesky factor
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, fall back to eigen
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                
                x = x_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                x = np.clip(x, -5.0, 5.0)
                
                gen += 1
                
                # Stagnation check
                if best_f < best_f_at_last_improve - 1e-6:
                    best_f_at_last_improve = best_f
                    last_improvement_gen = gen
                
                # If no improvement for many generations, break to restart
                if gen - last_improvement_gen > 15:
                    break
                
                # If sigma is very small, likely converged
                if sigma < 1e-8:
                    break
            
            return best_x, best_f, call_count
        
        # Multi-start strategy with adaptive restarts
        # Phase 1: Initial CMA-ES from origin with moderate sigma
        alloc1 = min(150, budget)
        run_cmaes(np.zeros(n), 2.0, alloc1)
        
        if not time_ok() or call_count >= budget:
            return best_x, best_f
        
        # Phase 2: Restart from best found, with smaller sigma for exploitation
        # or from a random point if we suspect we're stuck
        # Check if we made good progress
        progress = best_f  # We don't know f_opt, so use relative heuristics
        
        # Try a few different strategies for restarts
        n_restarts = 3
        remaining = budget - call_count
        if remaining > 30:
            # Divide remaining budget among restarts
            per_restart = remaining // (n_restarts + 1)
            
            for r in range(n_restarts):
                if not time_ok() or call_count >= budget:
                    break
                
                # Choose restart strategy
                if r == 0:
                    # Exploit: start from best, small sigma
                    x0 = best_x.copy()
                    sigma0 = 0.3
                elif r == 1:
                    # Explore: random point
                    x0 = rng.uniform(-4.0, 4.0, n)
                    sigma0 = 1.5
                else:
                    # Mixed: perturb best
                    x0 = best_x + rng.normal(0, 0.5, n)
                    x0 = np.clip(x0, -5.0, 5.0)
                    sigma0 = 0.8
                
                max_calls = per_restart
                run_cmaes(x0, sigma0, max_calls)
            
            # Use any remaining budget for final exploitation
            if time_ok() and call_count < budget:
                run_cmaes(best_x, 0.1, budget - call_count)
        
        return best_x, best_f
