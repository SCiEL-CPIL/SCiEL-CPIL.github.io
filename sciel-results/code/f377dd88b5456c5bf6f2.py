import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        # Strict wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f # Return best known if budget exceeded (shouldn't happen if logic is correct)
            
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            
            calls_made += 1
            try:
                f_val = func(x_clipped)
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Hand-rolled CMA-ES
        # Parameters tuned for fast convergence on Sphere
        def cma_es(x0, sigma0, maxfvals, seed):
            nonlocal calls_made, best_f, best_x
            
            if calls_made + 10 > budget:
                return
            
            rng = np.random.RandomState(seed)
            
            n = dim
            m = max(4 + int(3 * np.log(n)), 12)  # Population size
            
            # CMA-ES parameters
            cm = 1.0 / (n + 4)
            cs = (n + 1.5) / (n + 9)
            damps = 0.5 + 1.0 / (n + 3)
            mu = m // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            c1 = (2 / ((n + 1.8) ** 2 + mueff)) * (cm - 1) + 2 / (n ** 1.5 + 10)
            cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            sigmapath = np.sqrt((1 - c1) / (1 + c1))
            
            # State
            mean = x0.copy()
            C = np.eye(n)
            sqrtC = np.eye(n)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            invsqrtC = np.eye(n)
            
            # Initial evaluation
            f_mean = counted_func(mean)
            
            # Loop
            while calls_made < budget:
                # Stopping criteria
                if sigma < 1e-12:
                    break
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cc) ** (2 * (calls_made // m))) > np.sqrt(n) * 1.5:
                    break
                # Check for stagnation in best value
                # If best_f hasn't improved in last 50 evals and sigma is small, break?
                # For now, rely on sigma reduction and maxfvals.
                
                # Sample population
                z = rng.randn(m, n)
                y = z @ sqrtC
                x = mean + sigma * y.T
                f = np.array([counted_func(xi) for xi in x])
                
                # Sort by fitness
                idx = np.argsort(f)
                y = y[idx]
                f = f[idx]
                
                # Update mean
                mean_old = mean.copy()
                mean = mean + sigma * np.sum(weights[:, None] * y, axis=0)
                
                # Update covariance matrix
                yw = (y - mean_old / sigma)
                C = (1 + c1 * cc * (np.outer(pc, pc) - C)) * (1 - c1) + \
                    cmu * np.sum(weights[:, None] * np.outer(yw.T, yw.T), axis=0)
                
                # Update path evolution strategies
                pc = (1 - cs) * pc + cs * np.sqrt(ccs := (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** (2 * m))) * (mean - mean_old) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean - mean_old) / sigma
                
                # Sigma update (Hessian update)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // m)))) / np.sqrt(n) < (1.4 + 2 / (n + 1))
                sigma *= np.exp((cs / damps) * (hsig * np.linalg.norm(ps) / np.sqrt(n) - 1))
                
                # Ensure positive definiteness and update sqrtC
                # Simple Cholesky with regularization
                try:
                    L = np.linalg.cholesky(C)
                    sqrtC = L
                    invsqrtC = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Regularize
                    C += 1e-4 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(C)
                        sqrtC = L
                        invsqrtC = np.linalg.inv(L)
                    except np.linalg.LinAlgError:
                        sqrtC = np.eye(n)
                        invsqrtC = np.eye(n)
                        C = np.eye(n)
                        sigma *= 0.5 # Aggressive shrink on failure
            
            # Final evaluation of best point found during CMA-ES run?
            # The wrapper already updates best_x/best_f.

        # Multi-start strategy
        # We want to converge FAST. 
        # Start 1: Large sigma, wide search.
        # If it stalls or takes too long, restart with smaller sigma and tighter bounds around current best.
        
        starts = [
            {'x0': np.zeros(dim), 'sigma0': 4.0, 'seed': 42, 'budget_frac': 0.4},
            {'x0': np.zeros(dim), 'sigma0': 1.0, 'seed': 123, 'budget_frac': 0.3},
            {'x0': np.zeros(dim), 'sigma0': 0.2, 'seed': 999, 'budget_frac': 0.2},
            {'x0': np.zeros(dim), 'sigma0': 0.05, 'seed': 555, 'budget_frac': 0.1},
        ]
        
        for i, start_info in enumerate(starts):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break
                
            # If we are already very close to a local minimum (assuming sphere, this is global),
            # we can skip to finer steps.
            
            x0 = start_info['x0']
            # If we have a good best_x, start near it for subsequent runs
            if i > 0 and np.isfinite(best_f):
                # Perturb slightly around best
                x0 = best_x + np.random.RandomState(start_info['seed']).randn(dim) * start_info['sigma0'] * 0.1
            
            sigma0 = start_info['sigma0']
            seed = start_info['seed']
            
            # Allocate budget for this run
            # Don't use all remaining, leave some for next
            run_budget = int(budget * start_info['budget_frac'])
            run_budget = min(run_budget, budget - calls_made)
            
            if run_budget < 10:
                continue
                
            # Run CMA-ES
            cma_es(x0, sigma0, run_budget, seed)
            
            # Check if we should stop early
            # If best_f is very small (relative to initial?), we might be done.
            # But we don't know f_opt. 
            # However, if sigma has collapsed, CMA-ES stops.
            
            # Heuristic: If we improved significantly in the last run, continue.
            # If not, try next start.

        return best_x, best_f
