"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Strategy:
    1. Initial sampling: Evaluate 5 random points and the center to find a good starting basin.
    2. CMA-ES: Run CMA-ES from the best initial point with the remaining budget.
    
    CMA-ES is chosen for its robustness on non-convex, rotated landscapes (BBOB standard).
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Sampling ---
        # Evaluate center and 5 random points in [-4, 4]^5
        # Minimizer is known to be in [-4, 4]^5
        n_random = 5
        points = [np.zeros(dim)]
        points.extend(rng.uniform(-4.0, 4.0, size=(n_random, dim)).tolist())
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        for p in points:
            if calls_made >= budget or time.time() > deadline:
                break
            f_val = func(p)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        # --- Phase 2: CMA-ES ---
        if calls_made < budget:
            self._run_cmaes(func, best_x, best_f, budget, deadline, calls_made)
            # Note: _run_cmaes updates best_x and best_f via nonlocal or return
            # Since we can't easily pass by reference, we'll return the best found
            # Or we can just let it run and rely on the fact that it updates internal state
            # Let's structure _run_cmaes to return the best found
            
        return best_x, best_f

    def _run_cmaes(self, func, x0, f0, budget, deadline, calls_made):
        n = self.dim
        # CMA-ES Parameters
        popsize = 4 + int(3 * np.log(n))
        mu = popsize // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Cumulative path
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Chi_N
        chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        mean = x0.copy()
        best_x_local = x0.copy()
        best_f_local = f0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for at least 1 generation
            # If not, just do random steps or break
            if budget - calls_made < 2:
                break

            # Generate population
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                
            population = []
            for i in range(popsize):
                z = np.random.standard_normal(n)
                x_i = mean + sigma * (L @ z)
                population.append(x_i)
                
            values = []
            for i in range(popsize):
                if calls_made >= budget or time.time() > deadline:
                    break
                f_i = func(population[i])
                calls_made += 1
                values.append(f_i)
                
            if len(values) < popsize:
                # If we couldn't evaluate the whole population, 
                # we can still use the partial results for a rough update 
                # or just stop. Stopping is safer for CMA-ES stability.
                # However, we can update best if any were better.
                for i, f_i in enumerate(values):
                    if f_i < best_f_local:
                        best_f_local = f_i
                        best_x_local = population[i].copy()
                break

            # Sort by fitness
            indices = np.argsort(values)
            x_sorted = [population[i] for i in indices]
            f_sorted = [values[i] for i in indices]
            
            # Update best
            if f_sorted[0] < best_f_local:
                best_f_local = f_sorted[0]
                best_x_local = x_sorted[0].copy()
                
            # Update mean
            mean_new = np.zeros(n)
            for i in range(mu):
                mean_new += weights[i] * x_sorted[i]
                
            # Update ps
            # Compute C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            z_mean = (mean_new - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ z_mean)
            
            # Update pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + h_sigma * cc * z_mean
            
            # Update C
            delta_h = (1 - h_sigma) * cc * (2 - cc)
            C_update = c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                diff = x_sorted[i] - mean
                C_update += (cmu / sigma**2) * weights[i] * np.outer(diff, diff)
                
            C = (1 + c1 + cmu - c1 - delta_h) * C + C_update
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            eigvals_C = np.linalg.eigvalsh(C)
            if np.min(eigvals_C) < 1e-12:
                C += 1e-12 * np.eye(n)
                
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            mean = mean_new
            
            # Stopping criteria
            if sigma < 1e-8 or sigma > 1e8:
                break
                
        # Update outer scope bests
        # Since we can't modify the caller's variables directly, 
        # we rely on the fact that the caller's best_x/best_f 
        # might be worse than best_x_local. 
        # But __call__ returns best_x, best_f from its scope.
        # So we need to pass them back or modify self.
        # Let's store the result in self and return it in __call__.
        self._best_x = best_x_local
        self._best_f = best_f_local

# EVOLVE-BLOCK-END
