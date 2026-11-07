import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        n = self.dim
        budget = self.budget
        
        # Track global best
        best_x = None
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Helper: Run CMA-ES for a given number of evaluations
        # Returns (best_x, best_f, remaining_budget)
        def run_cmaes(func, x0, sigma0, max_calls):
            nonlocal best_x, best_f
            calls_made = 0
            
            # Parameters
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
            
            # Evaluate initial point
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            while calls_made < max_calls:
                if time.time() > deadline:
                    break
                
                # Generate population
                B = rng.standard_normal((lam, n))
                
                # Update mean and step size
                # Sample
                # C^{1/2} via eigen decomposition (stable for small n)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                Y = x + sigma * (B @ sqrt_C.T)
                
                # Evaluate
                F = []
                for i in range(lam):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    y = np.clip(Y[i], -5, 5)
                    fy = func(y)
                    F.append(fy)
                    calls_made += 1
                    update_best(y, fy)
                
                if len(F) < mu:
                    break
                
                F_arr = np.array(F)
                Y_arr = np.array(Y[:len(F)])
                
                # Sort
                idx = np.argsort(F_arr)
                Y_sorted = Y_arr[idx]
                F_sorted = F_arr[idx]
                
                # Update mean
                x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update step size
                y_invC = (Y_sorted[:mu] - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))
                
                # Update correlation matrix
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                # Update covariance
                # h_pc check
                norm_ps = np.linalg.norm(ps)
                norm_ps_expected = np.sqrt(n) * (1 - (1 - c_sigma)**(0.5)) # Approx
                # Standard check
                delta_h = 1 - np.exp(-2 + 5 / (n + 1))
                # Use generation counter approximation
                gen = calls_made // lam
                if norm_ps / np.sqrt(1 - (1 - c_sigma)**((gen + 1) / d_sigma)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n:
                    h_pc = delta_h
                else:
                    h_pc = 0.0
                    
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    y_i = y_invC[i]
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                C = (C + C.T) / 2
                
                x = x_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Clip x to box
                x = np.clip(x, -5, 5)
                
                # Stagnation check: if sigma is very small or no improvement in recent history
                # For simplicity, we rely on the outer restart loop.
                
            return best_x, best_f, calls_made
        
        # Strategy:
        # 1. Try CMA-ES from origin.
        # 2. If it converges (sigma small) or stagnates, restart from a random point or best found.
        # Given 500 evals, we can afford a few restarts.
        
        # Phase 1: CMA-ES from 0
        # Use ~200 evals
        alloc1 = 200
        best_x, best_f, used1 = run_cmaes(func, np.zeros(n), 1.0, alloc1)
        
        if time.time() > deadline or used1 >= budget:
            return best_x, best_f
        
        # Check for convergence/stagnation
        # If best_f is very small, we might be done. But we don't know f_opt.
        # If sigma collapsed, we should restart.
        # Heuristic: If the last 50 evals didn't improve best_f significantly, restart.
        # We don't track history easily here, so let's just do a second run from the best point with a perturbed sigma,
        # or a random point if the first one seems stuck.
        
        # Phase 2: CMA-ES from best_x found so far, but with a fresh sigma (maybe smaller if we converged, larger if not?)
        # Actually, if we converged, sigma is small. If we just ran out of budget, sigma is whatever.
        # Let's try a restart from a random point to escape local minima (BBOB f6 is multimodal? "Attractive sector" suggests it might be unimodal but with a sector structure).
        # f6 is "Attractive sector". It is likely unimodal but anisotropic.
        # CMA-ES should handle anisotropy well.
        
        # Let's use remaining budget for a second CMA-ES run starting from the current best, but resetting the covariance?
        # Or just continue? The function `run_cmaes` resets state.
        # Let's start a new CMA-ES from the best point found, with sigma = 0.1 * (box_size / 10)?
        # Or better: If the first run converged (sigma < 1e-6), we are likely at the minimum.
        # If not, we might be stuck.
        
        # Let's check if we improved significantly.
        # Simple heuristic: Run CMA-ES again from a random point if the first one didn't get close to 0?
        # We don't know 0.
        
        # Let's just use the remaining budget for another CMA-ES run starting from the best point found,
        # but initializing the covariance to Identity and sigma to a moderate value.
        # This effectively "restarts" the adaptation while keeping the mean near the best solution.
        
        alloc2 = budget - used1
        if alloc2 > 10:
            # Start from best_x, but with a fresh sigma
            # If we are close to the optimum, sigma should be small.
            # Let's try sigma = 0.1
            best_x, best_f, used2 = run_cmaes(func, best_x, 0.5, alloc2)
            
        return best_x, best_f
