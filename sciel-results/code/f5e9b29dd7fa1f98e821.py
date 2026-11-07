"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        def time_left():
            return deadline - time.time()

        def run_cmaes(func, x0, f0, max_calls, dim, rng, deadline):
            # CMA-ES parameters
            # lambda: population size
            # mu: number of best points selected
            # For dim=5, lambda=9 or 10 is standard. Let's use 10 for diversity.
            lam = max(6, 4 + int(3 * math.log(dim))) 
            if lam % 2 == 0:
                lam += 1 # Keep odd? Not strictly necessary but standard in some impls. Let's just use 9 or 10.
            # Let's stick to standard: lam = 4 + floor(3*log(dim)) = 4 + 3*1.6 = 8.8 -> 9
            lam = 9
            mu = lam // 2 # 4
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1 / np.sum(w**2)
            
            # Step size control parameters
            cs = (mueff + 1.5) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - cs
            sigma0 = 1.0 # Initial step size. Box is [-5,5], center 0.
            
            # Covariance adaptation parameters
            c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2)**2 + mueff))
            c_sigma = (mueff + 1.5) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - c_sigma
            
            # State
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            calls_used = 1 # x0 is already evaluated
            best_x = x0.copy()
            best_f = f0
            
            # Main loop
            while calls_used < max_calls and time_left() > 0.1:
                # Generate population
                pop = np.zeros((lam, dim))
                fit = np.zeros(lam)
                
                for i in range(lam):
                    # Sample
                    z = rng.standard_normal(dim)
                    # Transform
                    # We need C^{1/2}. For dim=5, we can use eigendecomposition or Cholesky.
                    # Cholesky is faster if C is positive definite.
                    # To ensure PD, we can add small jitter if needed, but CMA-ES usually maintains PD.
                    try:
                        L = np.linalg.cholesky(C)
                        y = L @ z
                    except np.linalg.LinAlgError:
                        # Fallback to eig if Cholesky fails
                        vals, vecs = np.linalg.eigh(C)
                        vals = np.clip(vals, 1e-12, None)
                        L = vecs @ np.diag(np.sqrt(vals))
                        y = L @ z
                        
                    x_new = x_mean + sigma * y
                    # Clip to box? CMA-ES usually works on unbounded, but task says func clips.
                    # It's better to let CMA-ES explore and let func clip, OR clip x_new to box.
                    # The task says: "func(x) clips a finite length-5 vector to the box".
                    # So we can pass unbounded x, func will clip.
                    # However, if we want to be efficient, we might want to stay near the box.
                    # But CMA-ES adapts. Let's pass unbounded.
                    
                    f_val = func(x_new)
                    pop[i] = x_new
                    fit[i] = f_val
                    calls_used += 1
                    
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x_new.copy()
                        
                # Sort by fitness
                order = np.argsort(fit)
                pop = pop[order]
                fit = fit[order]
                
                # Update mean
                x_new_mean = np.sum(pop[:mu, :] * w[:, None], axis=0)
                
                # Update step size
                # Need C^{-1/2} for pc update
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.clip(vals, 1e-12, None)
                    C_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals))
                    
                pc = (1 - cs) * pc + cs * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
                norm_pc = np.linalg.norm(pc)
                
                hsig = (norm_pc / math.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chiN)
                
                ps = (1 - cs) * ps + hsig * math.sqrt(cs * (2 - cs) * mueff) / sigma * (x_new_mean - x_mean)
                norm_ps = np.linalg.norm(ps)
                
                sigma *= math.exp(cs / d_sigma * (norm_ps / chiN - 1))
                
                # Update covariance
                rank_one = np.outer(ps, ps)
                rank_mu = np.sum(w[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu))
                
                C = (1 + c1 * (2 - c1) - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * (1 - c1) * C) + cmu * rank_mu
                
                # Re-normalize C to prevent drift in trace
                # Actually, standard CMA-ES doesn't explicitly re-normalize trace every step, 
                # but the update formula preserves the expected trace if parameters are correct.
                # However, numerical errors can accumulate.
                # Let's ensure symmetry
                C = (C + C.T) / 2
                
                # Update mean
                x_mean = x_new_mean
                
                # Optional: Restart if sigma is too small or large
                # If sigma < 1e-12 or sigma > 1e6, break or restart.
                if sigma < 1e-12:
                    break
                if sigma > 1e6:
                    break

            return best_x, best_f, calls_used

        # Strategy:
        # 1. Start with a reasonable initial point. The optimum is in [-4,4]^5.
        #    Let's start at 0,0,0,0,0 or a random point?
        #    Random start is safer for multi-modal.
        # 2. Run CMA-ES with the full budget?
        #    CMA-ES is global-ish but can get stuck.
        #    Let's do a few restarts if it stagnates?
        #    With budget 500 and lam=9, we get ~55 generations.
        #    That might be enough for dim=5.
        #    
        #    Let's try a single run first, but with a random start.
        #    Actually, let's do 2 runs if time allows?
        #    No, CMA-ES is expensive per generation.
        #    Let's just run one CMA-ES with a random start.
        
        # Initial point
        x0 = rng.uniform(-4, 4, dim)
        f0 = func(x0)
        calls_made = 1
        
        best_x, best_f, used = run_cmaes(func, x0, f0, budget - calls_made, dim, rng, deadline)
        calls_made += used
        
        # If we have budget left, try another random start?
        # CMA-ES usually uses most of the budget.
        # If it terminated early due to convergence, we might have budget left.
        if calls_made < budget and time_left() > 1.0:
            # Try a second run from a different random start
            x0_2 = rng.uniform(-4, 4, dim)
            f0_2 = func(x0_2)
            calls_made += 1
            if f0_2 < best_f:
                best_f = f0_2
                best_x = x0_2.copy()
                
            if calls_made < budget and time_left() > 1.0:
                best_x_2, best_f_2, used_2 = run_cmaes(func, x0_2, f0_2, budget - calls_made, dim, rng, deadline)
                calls_made += used_2
                if best_f_2 < best_f:
                    best_f = best_f_2
                    best_x = best_x_2.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
