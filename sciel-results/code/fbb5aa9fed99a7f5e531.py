"""CMA-ES + Basin-Hopping + Nelder-Mead + L-BFGS-B hybrid optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    CMA-ES + Basin-Hopping + Nelder-Mead + L-BFGS-B hybrid optimizer.
    
    Strategy:
    1. Phase 1: Global exploration using CMA-ES with tuned parameters.
       CMA-ES is robust to multimodality and rotation, maintaining a population
       that explores the landscape efficiently while adapting the covariance.
    2. Phase 2a: Basin-Hopping to escape local minima of Schaffers F7.
       Performs stochastic jumps followed by aggressive Nelder-Mead refinements.
    3. Phase 2b: L-BFGS-B for high-precision polish.
    4. Phase 2c: Final tight-tolerance Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Clip to valid range
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            
            try:
                fx = float(func(x))
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls += 1
                return np.inf

        # --- Phase 1: CMA-ES ---
        # Parameters tuned for 5D
        lambda_ = 12  # Larger population for better exploration in 5D
        mu = max(1, int(lambda_ / 2))
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Initial values
        mean = np.zeros(n)
        C = np.eye(n)
        sigma = 2.0  # Initial step size for 5D [-5,5] search
        
        # Adaptation rates
        c_sigma = (1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + mu_eff / (n + 1.0)) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu / n)
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0)**2 + mu_eff)
        
        # Paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial evaluation of the mean
        f0 = evaluate(mean)
        
        # Main CMA-ES Loop
        # Reserve about 30% of the budget for local search
        cma_budget = int(budget * 0.70)
        
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        while calls < cma_budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            Z = rng.standard_normal((lambda_, n))
            X = mean + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.array([evaluate(x) for x in X])
            
            if calls >= cma_budget:
                break
                
            # Sort
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            if np.isinf(F_sorted[0]):
                break
                
            # Update mean
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Standardize
            v = (mean_new - mean) / sigma
            try:
                y_p = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y_p = v
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * y_p
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for E||N(0,I)||
            E_norm = np.sqrt(n) * (1.0 - 1.0/(4.0*n) + 7.0/(32.0*n**2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1.0))
            sigma = np.clip(sigma, 1e-12, 5.0)
            
            # Update p_c
            # C^-1/2 v
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                y_c = C_inv_half @ v
            except np.linalg.LinAlgError:
                y_c = v
                
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * y_c
            
            # Update C
            c_mu_adapt = c_mu * (1.0 - c_sigma / d_sigma * (norm_p_sigma**2 - 1.0))
            c_mu_adapt = np.clip(c_mu_adapt, 0.0, 2.0 / mu_eff)
            
            Y_top = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y_top.T).T
            except np.linalg.LinAlgError:
                Y_std = Y_top
                
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            C = (1 - c_mu_adapt) * C + c_mu_adapt * outer_sum
            C = (C + C.T) / 2.0
            
            mean = mean_new
            
            # Convergence check
            if sigma < 1e-7:
                break

        # --- Phase 2: Multi-method Local Refinement ---
        # If we still have budget and found a finite best point
        if not np.isinf(best_f) and calls < budget and time.time() < deadline:
            
            # Phase 2a: Basin-Hopping to escape local minima
            # Schaffers F7 has many local minima; basin hopping helps find the global one
            try:
                remaining = budget - calls
                # Use a portion of remaining budget for basin hopping
                bh_budget = int(remaining * 0.4)
                if bh_budget > 20:
                    current_x = best_x.copy()
                    current_f = best_f
                    bh_rng = np.random.default_rng(123)
                    
                    # Basin hopping parameters
                    n_bh_steps = min(20, bh_budget // 10)
                    jump_scale = 1.0  # Initial jump scale
                    local_max_iter = 20  # Max iterations for local refinement per step
                    
                    for step in range(n_bh_steps):
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        # Stochastic jump
                        jump = bh_rng.standard_normal(n) * jump_scale
                        new_x = np.clip(current_x + jump, -5.0, 5.0)
                        
                        # Local refinement from new point using Nelder-Mead
                        try:
                            res_local = minimize(
                                lambda x: evaluate(x),
                                new_x,
                                method='Nelder-Mead',
                                options={'maxiter': local_max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex_size': 0.3}
                            )
                            if res_local.fun < current_f:
                                current_f = res_local.fun
                                current_x = res_local.x.copy()
                                # Accept: update best
                                if current_f < best_f:
                                    best_f = current_f
                                    best_x = current_x.copy()
                                # Reduce jump scale on success
                                jump_scale *= 0.7
                            else:
                                # Reject: keep current, increase jump scale slightly
                                jump_scale = min(jump_scale * 1.2, 3.0)
                        except Exception:
                            pass
                        
                        # Adapt jump scale based on progress
                        if step > 0 and step % 5 == 0:
                            # If no improvement in last 5 steps, increase exploration
                            jump_scale = min(jump_scale * 1.5, 4.0)
                    
                    # After basin hopping, update best
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
            except Exception:
                pass
            
            # Phase 2b: Nelder-Mead for robust local search from best point
            if not np.isinf(best_f) and calls < budget and time.time() < deadline:
                try:
                    remaining = budget - calls
                    nm_max_iter = max(30, int(remaining * 0.3 / max(1, n)))
                    
                    res_nm = minimize(
                        lambda x: evaluate(x), 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': nm_max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'initial_simplex_size': 0.5}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass
            
            # Phase 2c: L-BFGS-B for high-precision polish (if budget remains)
            if not np.isinf(best_f) and calls < budget and time.time() < deadline:
                try:
                    remaining = budget - calls
                    lbfgs_max_iter = max(10, int(remaining * 0.4 / max(1, n)))
                    
                    res_lbfgs = minimize(
                        lambda x: evaluate(x), 
                        best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': lbfgs_max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
                    )
                    if res_lbfgs.fun < best_f:
                        best_f = res_lbfgs.fun
                        best_x = res_lbfgs.x.copy()
                except Exception:
                    pass
            
            # Phase 2d: Final tight-tolerance Nelder-Mead polish
            if not np.isinf(best_f) and calls < budget and time.time() < deadline:
                try:
                    remaining = budget - calls
                    if remaining > 10:
                        nm_polish_iter = max(20, int(remaining * 0.5 / max(1, n)))
                        
                        res_polish = minimize(
                            lambda x: evaluate(x), 
                            best_x, 
                            method='Nelder-Mead', 
                            options={'maxiter': nm_polish_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex_size': 0.1}
                        )
                        if res_polish.fun < best_f:
                            best_f = res_polish.fun
                            best_x = res_polish.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
