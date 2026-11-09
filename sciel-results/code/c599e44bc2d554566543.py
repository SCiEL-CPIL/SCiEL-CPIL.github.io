import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda=10 is a good standard choice
        lam = 10 
        mu = lam // 2
        
        # Weights for the mean update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2 + mueff / (dim + 1.3))
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        c_sigma = (mueff + 1.5) / (dim + mueff + 5)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def run_cmaes(start_x, start_sigma, max_calls):
            calls = 0
            m = start_x.copy()
            sigma = start_sigma
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            best_x = m.copy()
            best_f = func(best_x)
            calls += 1
            
            # Initial population generation
            B = np.linalg.inv(np.linalg.cholesky(C))
            D = np.sqrt(np.diag(C))
            
            no_improve_count = 0
            last_best_f = best_f
            
            while calls < max_calls:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = np.zeros((lam, dim))
                for i in range(lam):
                    z = np.random.randn(dim)
                    pop[i] = m + sigma * (B @ z)
                
                # Evaluate
                fit = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    x_i = np.clip(pop[i], -5.0, 5.0)
                    fit[i] = func(x_i)
                    calls += 1
                    if fit[i] < best_f:
                        best_f = fit[i]
                        best_x = x_i.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                # If we hit the deadline or call limit, break
                if calls >= max_calls or time.time() > deadline:
                    break
                
                # Sort by fitness
                sort_idx = np.argsort(fit)
                y = np.copy(pop[sort_idx])
                
                # Update mean
                y_w = weights[:, None] * y[:mu]
                m_new = m + sigma * (y_w.sum(axis=0) / mueff)
                
                # Update covariance matrix
                y_inv = (y[:mu] - m) / sigma
                C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - cs) * np.eye(dim)) + \
                        cmu * (y_w.T @ y_w) / mueff
                
                # Symmetrize
                C_new = (C_new + C_new.T) / 2
                
                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((y_w.sum(axis=0) / mueff) / sigma))
                # Note: The standard CMA-ES update for ps uses the transformed step
                # Let's use a simpler, more robust sigma update to avoid implementation errors
                # Standard CMA-ES sigma update:
                # sigma = sigma * exp(cs / ds * (||ps|| / chi_n - 1))
                
                # Recalculate B and D for next iteration
                try:
                    # Ensure C is positive definite
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    if np.min(eigvals) < 1e-10:
                        C_new = C # Fallback
                    else:
                        C = C_new
                        B = np.linalg.inv(np.linalg.cholesky(C))
                except:
                    C = np.eye(dim)
                    B = np.eye(dim)
                
                # Update pc
                pc = (1 - cs) * pc + cs * (y_w.sum(axis=0) / mueff)
                
                # Update sigma
                # We need the norm of the step in the transformed space
                # step = (m_new - m) / sigma
                # But we used y_w which is already weighted and scaled?
                # Let's stick to the standard formula:
                # d = (m_new - m) / sigma
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B @ d)
                d = (m_new - m) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ d)
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                
                m = m_new
                
                # Restart logic
                if no_improve_count > 15: # 15 generations * 10 pop = 150 calls without improvement
                    # If stuck, restart with a larger sigma around best
                    if sigma < 0.5:
                         return best_x, best_f, calls
                    # Or just break and let the outer loop restart
                    break

            return best_x, best_f, calls

        # Outer loop: Multiple restarts
        global_best_x = np.random.uniform(-4, 4, dim)
        global_best_f = func(global_best_x)
        total_calls = 1
        
        rng = np.random.default_rng(42)
        
        # We will allocate budget to several independent CMA-ES runs
        # Since the landscape has a plateau, one run might miss the basin.
        # We'll do a few short runs.
        
        num_restarts = 5
        calls_per_restart = (budget - total_calls) // num_restarts
        
        for r in range(num_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize start point
            if r == 0:
                start_x = global_best_x.copy()
                start_sigma = 2.0
            else:
                # Random restart, but biased towards the center or best
                if rng.random() < 0.5:
                    start_x = rng.uniform(-4, 4, dim)
                else:
                    start_x = global_best_x + rng.uniform(-2, 2, dim)
                    start_x = np.clip(start_x, -5, 5)
                start_sigma = 1.5
            
            max_calls_this_run = min(calls_per_restart, budget - total_calls)
            
            if max_calls_this_run < 20:
                break
                
            x, f, c = run_cmaes(start_x, start_sigma, max_calls_this_run)
            total_calls += c
            
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
                
            # If we found a very good solution, stop early?
            # The plateau value is usually significantly higher than the optimum.
            # If we are close to the optimum, we might want to refine.
            # But CMA-ES handles refinement.
            
        return global_best_x, global_best_f
