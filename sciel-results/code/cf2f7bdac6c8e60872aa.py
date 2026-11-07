"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Hybrid optimizer: CMA-ES for global exploration, followed by a Nelder-Mead
    polish phase for local convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: CMA-ES ---
        # Use a significant portion of the budget for CMA-ES to handle anisotropy
        # and find the basin of attraction.
        # For n=5, lambda=9, mu=4 is standard.
        alloc1 = int(budget * 0.7)  # 70% for CMA-ES
        if alloc1 < 20:
            alloc1 = budget
        
        x = np.zeros(n)
        sigma = 2.0  # Initial sigma, reasonable for [-5,5]
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # CMA-ES parameters
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
        
        calls_made = 0
        
        # Evaluate initial point
        fx = func(x)
        calls_made += 1
        update_best(x, fx)
        
        while calls_made < alloc1:
            if time.time() > deadline:
                break
            
            # Generate population
            B = rng.standard_normal((lam, n))
            
            # Compute C^{1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Y = x + sigma * (B @ sqrt_C.T)
            
            # Evaluate population
            F = []
            Y_eval = []
            for i in range(lam):
                if calls_made >= alloc1 or time.time() > deadline:
                    break
                y = np.clip(Y[i], -5, 5)
                fy = func(y)
                F.append(fy)
                Y_eval.append(y)
                calls_made += 1
                update_best(y, fy)
            
            if len(F) < mu:
                break
                
            F_arr = np.array(F)
            Y_arr = np.array(Y_eval)
            
            # Sort
            idx = np.argsort(F_arr)
            Y_sorted = Y_arr[idx]
            
            # Update mean
            x_new = np.sum(w[:mu, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update step size
            y_invC = (Y_sorted[:mu] - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (w[:mu] @ y_invC))
            
            # Update correlation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
            
            # Update covariance
            norm_ps = np.linalg.norm(ps)
            # Simplified h_pc check
            h_pc = 0.0
            # A more robust check would use a moving average of norm(ps)/norm(ps_expected)
            # but for short runs, a simple threshold is often enough.
            # If ps is large, the mean shift is likely not due to noise.
            # Expected norm of ps is sqrt(n) * (1 - (1-c_sigma)^(1/2)) approx
            # Let's use the standard condition
            delta_h = 1 - np.exp(-2 + 5 / (n + 1))
            # We don't have a generation counter easily accessible in this loop structure without extra vars
            # Let's approximate: if norm_ps is significantly larger than expected, h_pc = delta_h
            # Expected norm is roughly sqrt(n) * sqrt(c_sigma * (2-c_sigma) * mu_eff) * sqrt(1/(1-(1-c_sigma)^(2/gen)))?
            # Simpler: if norm_ps > 1.4 * sqrt(n), assume drift.
            if norm_ps > 1.4 * np.sqrt(n):
                h_pc = delta_h
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
            for i in range(mu):
                y_i = y_invC[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            C = (C + C.T) / 2
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n) - 1))
            
            # Clip x to box
            x = np.clip(x, -5, 5)
            
            # Convergence check: if sigma is very small, break
            if sigma < 1e-10:
                break

        # --- Phase 2: Nelder-Mead Polish ---
        # Use the remaining budget for a local polish using Nelder-Mead.
        # This helps converge to the exact minimum if CMA-ES is close.
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0 and best_x is not None:
            # Initialize simplex
            # Start from best_x found by CMA-ES
            # Create a simplex around best_x
            # Use a small step size, e.g., 0.1 or based on sigma if available
            # Since we don't have access to final sigma easily, use a fixed small step
            step = 0.5
            
            simplex = [best_x.copy()]
            for i in range(n):
                pt = best_x.copy()
                # Perturb one dimension
                pt[i] += step
                simplex.append(pt)
            
            # Evaluate initial simplex
            fvals = []
            for i, pt in enumerate(simplex):
                if calls_made >= budget or time.time() > deadline:
                    break
                f = func(pt)
                fvals.append(f)
                calls_made += 1
                update_best(pt, f)
            
            if len(fvals) < n + 1:
                return best_x, best_f
                
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma_nm = 0.5 # Shrink
            
            # Main loop
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                # If the spread of function values is small, stop
                if max(fvals) - min(fvals) < 1e-10:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                update_best(x_r, f_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    update_best(x_e, f_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = func(x_c)
                        calls_made += 1
                        update_best(x_c, f_c)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = func(x_c)
                        calls_made += 1
                        update_best(x_c, f_c)
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made < budget and time.time() < deadline:
                                f_s = func(simplex[i])
                                calls_made += 1
                                update_best(simplex[i], f_s)
                                fvals[i] = f_s
                        if calls_made >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
