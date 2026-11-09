"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """
    Optimizer using a hand-rolled CMA-ES with restarts for BBOB f18 (Schaffers F7 ill-conditioned).
    
    CMA-ES adapts the search distribution to the local curvature and scaling, making it
    highly effective for ill-conditioned functions. We implement it from scratch to avoid
    dependency issues and to have full control over the budget.
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
        # lambda = 4 + floor(3 * ln(n))
        # For n=5, lambda = 4 + 3*1.609 = 8.8 -> 8 (floor) or 9 (round). Standard is 4+3ln(n).
        # Let's use lambda = 9 for n=5 to be safe and robust.
        lambda_pop = 4 + int(3 * math.log(dim))
        if lambda_pop < 4:
            lambda_pop = 4
            
        # Standard deviation
        sigma0 = 2.0
        
        # Weights
        mu = lambda_pop // 2
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
        
        # Pre-compute random matrix
        def sample_random():
            return np.random.randn(lambda_pop, dim)
        
        # Main loop with restarts
        while calls < total_budget:
            if time.time() > deadline - 5.0:
                break
                
            # Initialize CMA-ES state
            mean = np.zeros(dim)
            C = np.eye(dim)
            sigma = sigma0
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            chol = np.linalg.cholesky(C)
            
            # Track progress for restart decision
            best_f_in_run = float('inf')
            calls_in_run = 0
            prev_best_f = float('inf')
            no_improvement_calls = 0
            max_no_improvement = 100  # Restart if no improvement for 100 calls
            
            # Run CMA-ES until budget exhausted or restart triggered
            while calls < total_budget and calls_in_run < 200:  # Max 200 calls per run
                if time.time() > deadline - 2.0:
                    break
                    
                # Sample population
                z = sample_random()
                pop = np.array([mean + sigma * (z[i] @ chol) for i in range(lambda_pop)])
                
                # Evaluate population
                fvals = []
                for x in pop:
                    if calls >= total_budget:
                        break
                    if time.time() > deadline - 1.0:
                        fvals.append(float('inf'))
                        continue
                    try:
                        f = func(x)
                        calls += 1
                        calls_in_run += 1
                        fvals.append(f)
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        if f < best_f_in_run:
                            best_f_in_run = f
                            no_improvement_calls = 0
                        else:
                            no_improvement_calls += 1
                    except:
                        fvals.append(float('inf'))
                
                if len(fvals) < lambda_pop:
                    fvals.extend([float('inf')] * (lambda_pop - len(fvals)))
                
                # Sort population
                order = np.argsort(fvals)
                sorted_pop = pop[order]
                sorted_fvals = fvals[order]
                
                # Update mean
                new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # Check if we should stop
                if np.linalg.norm(chol) < 1e-15:
                    break
                
                # Update evolution paths
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(chol) * (chol.T @ (new_mean - mean) / sigma)
                
                hsig = 1 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**((calls + 1) / (cs * (dim + 1)))) < (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1)) else 0
                
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (new_mean - mean) / sigma
                
                # Update C
                C = (1 + c1 * hsig * (cc - cc + 1) + cmu - c1) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(sorted_pop[:mu] - mean, sorted_pop[:mu] - mean), axis=0) / (sigma**2)
                
                # Ensure C is positive definite
                try:
                    chol = np.linalg.cholesky(C + 1e-20 * np.eye(dim))
                except:
                    # If Cholesky fails, reset C to identity and sigma to 1
                    C = np.eye(dim)
                    chol = np.eye(dim)
                    sigma = 1.0
                    
                # Update sigma
                sigma = sigma * math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(mueff) - 1))
                
                # Update mean
                mean = new_mean
                
                # Clip mean to box to keep search within bounds
                mean = np.clip(mean, -5, 5)
                
                # Check for restart condition
                if no_improvement_calls > max_no_improvement and calls_in_run > 50:
                    break
            
            # If we still have budget, restart from a new random point near the best found so far
            if calls < total_budget:
                # Perturb the best point slightly to escape local minima
                best_x_perturbed = best_x + 0.5 * np.random.randn(dim)
                best_x_perturbed = np.clip(best_x_perturbed, -5, 5)
                # Try evaluating the perturbed point
                try:
                    f = func(best_x_perturbed)
                    calls += 1
                    if f < best_f:
                        best_f = f
                        best_x = best_x_perturbed.copy()
                except:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
