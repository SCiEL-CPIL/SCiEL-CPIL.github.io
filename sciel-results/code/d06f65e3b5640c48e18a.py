"""Optimized BBOB f14 solver with CMA-ES and Nelder-Mead Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with Nelder-Mead polish for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to safely evaluate function and update best
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES ---
        # CMA-ES is robust for global optimization and adapts to the shape of the function.
        
        # Initialize CMA-ES parameters
        # Start at center of the box, with a step size that covers a good portion of the box initially
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix starts as identity
        C = np.eye(dim)
        
        # Inverse square root of C, for sampling
        # We will update this periodically
        C_eigvals, C_eigvecs = np.linalg.eigh(C)
        C_eigvals = np.maximum(C_eigvals, 1e-10) # Numerical stability
        C_inv_sqrt = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
        
        # Step size control parameters
        d_sigma = 1 + 2 * max(0, (dim + 1) / 3 - 1) + 0.5 * (dim / (dim + 1) - 1)
        c_sigma = (mu_eff + 2) / ((dim + 1.3) * d_sigma + 1)
        
        # Covariance matrix adaptation parameters
        # mu is the number of best solutions used for recombination
        # lambda is the population size
        # For dim=5, lambda is typically around 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        # We need to evaluate the initial mean and the first population
        
        # Evaluate initial mean
        f_mean = evaluate(mean)
        
        # We will run generations until budget is low or time is up
        # Let's reserve 50 calls for the final polish phase
        polish_budget = 50
        cma_budget_limit = budget - polish_budget
        
        while calls_made < cma_budget_limit and time.time() < deadline:
            # Sample population
            population = []
            f_values = []
            
            for i in range(lam):
                # Sample from multivariate normal
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                # x = mean + sigma * C_inv_sqrt @ z
                x = mean + sigma * (C_inv_sqrt @ z)
                # Clip to box
                x = np.clip(x, -5, 5)
                
                fx = evaluate(x)
                if calls_made >= budget:
                    break
                population.append(x)
                f_values.append(fx)
            
            if len(population) < lam:
                break
                
            # Sort population by function value
            idx = np.argsort(f_values)
            population = [population[i] for i in idx]
            f_values = [f_values[i] for i in idx]
            
            # Update mean
            new_mean = np.sum([w[i] * population[i] for i in range(mu)], axis=0)
            
            # Update sigma control
            # old_norm = np.linalg.norm(mean) # Not quite, need norm of (new_mean - mean) / sigma
            # Actually, standard CMA-ES updates p_sigma based on the step
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((new_mean - mean) / sigma)
            
            step = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            # h_sigma check (simplified, often omitted in basic implementations or approximated)
            # For robustness, we just update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update C
            # h_c = 1 if norm(p_c) / sqrt(1 - (1-c_c)^(2*(gen+1))) < (1.4 + 2/(dim+1)) * sqrt((mu_eff-1)*(1-c_c)) else 0
            # Simplified: always update, or use a fixed threshold
            # For simplicity and speed, we'll do the standard rank-mu update
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i @ y_i.T))
            
            # y_i = (population[i] - mean) / sigma
            rank_mu_sum = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - mean) / sigma
                rank_mu_sum += w[i] * np.outer(y_i, y_i)
                
            # Approximate h_c as 1 for simplicity (or 0 if we want to be conservative)
            # Let's use 1 for now
            h_c = 1.0
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * rank_mu_sum
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigvals, 1e-10)
            C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
            C_inv_sqrt = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Check for convergence
            if sigma < 1e-8:
                break

        # --- Phase 2: Nelder-Mead Polish ---
        # Use the best point found by CMA-ES as the starting point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        # Allocate remaining budget to Nelder-Mead
        remaining_budget = budget - calls_made
        
        # Nelder-Mead parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma_nm = 0.5 # Shrink
        
        # Initialize simplex
        initial_step = 0.1 # Smaller step for polish
        
        simplex = []
        simplex_f = []
        
        simplex.append(best_x.copy())
        simplex_f.append(best_f)
        
        for i in range(dim):
            v = best_x.copy()
            v[i] += initial_step
            v = np.clip(v, -5, 5)
            fv = evaluate(v)
            simplex.append(v)
            simplex_f.append(fv)
            
            if calls_made >= budget:
                break
        
        # Main Nelder-Mead Loop
        while calls_made < budget and time.time() < deadline:
            # Sort vertices
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Check convergence
            if simplex_f[-1] - simplex_f[0] < 1e-8:
                break
            
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            x_second_worst = simplex[-2]
            f_second_worst = simplex_f[-2]
            x_best = simplex[0]
            f_best = simplex_f[0]
            
            x_c = np.sum(simplex[:-1], axis=0) / dim
            
            # Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < f_best:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < f_second_worst:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_co = x_c + 0.5 * (x_r - x_c)
                else:
                    x_co = x_c + 0.5 * (x_worst - x_c)
                
                x_co = np.clip(x_co, -5, 5)
                f_co = evaluate(x_co)
                
                if f_co < f_worst:
                    simplex[-1] = x_co
                    simplex_f[-1] = f_co
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = evaluate(simplex[i])
                        
                        if calls_made >= budget:
                            break
                    
                    if calls_made >= budget:
                        break

        # Ensure best_x is the absolute best found
        if best_x is None:
             best_x = np.zeros(dim)
             best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
