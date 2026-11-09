"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizer using a corrected CMA-ES with restarts for BBOB f18 (Schaffers F7 ill-conditioned).
    
    CMA-ES adapts the search distribution to the local curvature and scaling, making it
    highly effective for ill-conditioned functions. We implement it from scratch with
    proper covariance updates and a restart mechanism for robustness.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = max(4 + int(3 * math.log(dim)), 10)
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Update rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2)**2 + mueff))
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, math.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        last_improvement_call = 0
        
        def cmaes_run(budget_for_this_run, initial_mean=None, initial_sigma=2.0):
            nonlocal best_x, best_f, calls, last_improvement_call
            
            if initial_mean is None:
                mean = np.zeros(dim)
            else:
                mean = initial_mean.copy()
            
            sigma = initial_sigma
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            try:
                chol = np.linalg.cholesky(C)
            except:
                chol = np.eye(dim)
            
            local_best_f = float('inf')
            local_best_x = mean.copy()
            local_calls = 0
            no_improve_count = 0
            
            while local_calls < budget_for_this_run:
                if time.time() > deadline - 5.0:
                    break
                
                # Sample population
                z = np.random.randn(lambda_pop, dim)
                try:
                    pop = mean + sigma * (z @ chol.T)
                except:
                    pop = mean + sigma * z
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if local_calls >= budget_for_this_run:
                        break
                    if time.time() > deadline - 2.0:
                        fvals.append(float('inf'))
                        local_calls += 1
                        continue
                    try:
                        f = func(x)
                        calls += 1
                        local_calls += 1
                        fvals.append(f)
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            last_improvement_call = calls
                        if f < local_best_f:
                            local_best_f = f
                            local_best_x = x.copy()
                    except:
                        fvals.append(float('inf'))
                        local_calls += 1
                
                # Pad with inf if needed
                while len(fvals) < lambda_pop:
                    fvals.append(float('inf'))
                    local_calls += 1
                
                if local_calls >= budget_for_this_run:
                    break
                
                # Sort population
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = np.array(fvals)[order]
                
                # Check for improvement
                if sorted_fvals[0] < local_best_f - 1e-10:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stop if stuck
                if no_improve_count > 30 and local_calls > 50:
                    break
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Update evolution paths
                try:
                    norm_chol = np.linalg.norm(chol, 'fro')
                    if norm_chol < 1e-15:
                        break
                    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / norm_chol * (chol.T @ (new_mean - mean) / sigma)
                except:
                    break
                
                # hsig check
                try:
                    ps_norm = np.linalg.norm(ps)
                    expected_ps = math.sqrt(1 - (1 - cs)**(local_calls / (cs * (dim + 1))))
                    threshold = (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1)) if dim > 1 else 1.4
                    hsig = 1 if ps_norm / expected_ps < threshold else 0
                except:
                    hsig = 1
                
                # Update pc
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                
                # Update C
                delta = np.array(sorted_fvals[:mu])
                diff = (sorted_pop[:mu] - mean) / sigma
                rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(diff, diff), axis=0)
                C = (1 + c1 * hsig * (cc - cc + 1) + cmu - c1) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * rank_mu
                
                # Ensure positive definite
                try:
                    chol = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
                except:
                    # Reset if not positive definite
                    C = np.eye(dim)
                    chol = np.eye(dim)
                    sigma = max(sigma * 0.5, 0.1)
                
                # Update sigma
                try:
                    sigma = sigma * math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(mueff) - 1))
                except:
                    sigma = max(sigma * 0.9, 0.01)
                
                sigma = max(sigma, 1e-10)
                mean = new_mean
                mean = np.clip(mean, -5, 5)
            
            return local_best_x, local_best_f, local_calls
        
        # First CMA-ES run with most of the budget
        budget1 = int(total_budget * 0.7)
        x1, f1, used1 = cmaes_run(budget1)
        
        # If we improved significantly, do a second run from the best point with smaller sigma
        if best_f < float('inf') and calls < total_budget - 20 and time.time() < deadline - 10:
            budget2 = min(total_budget - calls, int(total_budget * 0.25))
            if budget2 > 10:
                x2, f2, used2 = cmaes_run(budget2, initial_mean=best_x, initial_sigma=0.5)
        
        # If still have budget and time, do a local refinement with small perturbations
        if calls < total_budget - 5 and time.time() < deadline - 5:
            remaining = total_budget - calls
            for _ in range(min(remaining, 20)):
                if time.time() > deadline - 2:
                    break
                # Small random perturbations around best
                for scale in [0.1, 0.05, 0.01]:
                    if calls >= total_budget:
                        break
                    x_try = best_x + scale * np.random.randn(dim)
                    x_try = np.clip(x_try, -5, 5)
                    try:
                        f_try = func(x_try)
                        calls += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                    except:
                        calls += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
