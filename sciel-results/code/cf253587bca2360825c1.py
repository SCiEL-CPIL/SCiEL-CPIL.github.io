"""
Hybrid CMA-ES / (1+1)-ES Optimizer for BBOB f11 (Discus).

Strategy:
1. Phase 1 (Exploration): Run a lightweight CMA-ES with a small population (lambda=10)
   for the first 20% of the budget. This helps navigate the rotated/shifted landscape
   and find the general basin of attraction without the high overhead of a full CMA-ES
   run.
2. Phase 2 (Exploitation): Switch to a (1+1)-ES with an aggressive step-size adaptation
   strategy. The (1+1)-ES is extremely efficient in 5D for smooth functions and can
   converge very fast once in the basin, which is critical for AOCC scoring (early 
   progress counts heavily).
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We aim to finish well before.
        deadline = start_time + 300
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            # The problem states func clips, but we clip here too for safety 
            # in our internal logic (e.g., simplex vertices)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        if best_x is None:
            return np.zeros(dim)

        # --- Phase 1: CMA-ES (Lightweight) ---
        # Use 20% of budget for CMA-ES
        cma_budget = int(budget * 0.2)
        
        # CMA-ES parameters
        lambda_pop = 10  # Small population for speed
        sigma = 2.0      # Initial step size
        mu = lambda_pop // 2
        # Weights: standard CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Covariance matrix initialization
        C = np.eye(dim)
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # State variables
        mean = best_x.copy()
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        sigma = 2.0
        
        # Pre-compute random directions? No, generate on fly for simplicity, 
        # but we need to be fast.
        
        cma_calls_used = 1 # We already evaluated x0
        
        while cma_calls_used < cma_budget and can_call():
            # Generate population
            B = np.random.normal(0, 1, (dim, lambda_pop))
            # Transform B with C^(1/2)
            # Since dim is small (5), we can do Cholesky
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)
            
            # Population points
            pop = np.array([mean + sigma * L @ B[:, i] for i in range(lambda_pop)])
            
            # Evaluate population
            f_pop = []
            for i in range(lambda_pop):
                fx = evaluate(pop[i])
                f_pop.append(fx)
                cma_calls_used += 1
                if not can_call():
                    break
            
            if len(f_pop) < lambda_pop:
                break
                
            f_pop = np.array(f_pop)
            # Sort indices by function value
            idx = np.argsort(f_pop)
            pop = pop[idx]
            f_pop = f_pop[idx]
            
            # Update mean
            new_mean = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (L.T @ (new_mean - mean))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (cma_calls_used))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)):
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (new_mean - mean) / sigma
            
            mean = new_mean
            
            # Update covariance
            delta_sig = 1.0
            # rank-mu update
            v = (pop[:mu] - mean) / sigma
            C = (1 + c1 * delta_sig + cmu) * C - c1 * (pc @ pc.T + delta_sig * C) - cmu * np.sum(w[:, None, None] * (v[:, :, None] * v[:, None, :]), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (cma_calls_used))) - np.sqrt(dim)))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Check convergence
            if np.linalg.norm(f_pop[:mu] - f_pop[mu:]) < 1e-12:
                break
        
        # --- Phase 2: (1+1)-ES ---
        # Use remaining budget
        # Start from current best
        x_cur = best_x.copy()
        f_cur = best_f
        
        # Adaptive step size for (1+1)-ES
        # Simple strategy: if improved, increase step slightly, else decrease.
        # Or use a more robust adaptation like in standard (1+1)-ES for BBOB.
        # Let's use a fast adaptation:
        # sigma *= exp(0.3 * (1 if improved else -1))
        
        # To maximize AOCC, we want to converge as fast as possible.
        # A slightly larger step size helps jump over small local minima in the 
        # "Discus" shape, but too large causes oscillation.
        
        sigma_es = 0.5
        last_improvement_step = 0
        
        while can_call():
            # Generate candidate
            d = np.random.normal(0, 1, dim)
            x_new = x_cur + sigma_es * d
            x_new = np.clip(x_new, -5, 5)
            
            f_new = evaluate(x_new)
            
            if f_new < f_cur:
                # Improvement
                x_cur = x_new
                f_cur = f_new
                # Increase step size to explore further if we are moving well
                # But don't grow too fast
                sigma_es *= 1.1
                last_improvement_step = calls_made
            else:
                # No improvement
                # Decrease step size to refine
                sigma_es *= 0.9
                
            # If we haven't improved for a long time, we might be stuck or converged.
            # In 5D, if sigma is very small and no improvement, we can stop or restart.
            # For AOCC, stopping early when converged is good if we can't improve.
            # But we must use budget? No, "repeating the last n_i if fewer calls are made".
            # So stopping early is fine if we are at the optimum.
            
            if sigma_es < 1e-6:
                # Considered converged
                # Try a few random restarts from best point with larger sigma?
                # Or just break.
                break
                
        return best_x

# EVOLVE-BLOCK-END
