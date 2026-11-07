"""Optimizer for BBOB-style f7 (Step ellipsoidal) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for step ellipsoidal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        call_count = 0

        def f(x):
            nonlocal call_count
            call_count += 1
            return func(np.asarray(x, dtype=float))

        best_x = None
        best_f = float('inf')

        # Phase 1: Coarse exploration with large sigma
        # Phase 2: Fine exploitation with adapted sigma
        # Use CMA-ES with restarts

        # CMA-ES parameters
        n = d  # dimension
        # Population size: 4 + floor(3 * ln(n))
        lambda_pop = max(4, 4 + int(3 * np.log(n)))
        
        # Standard CMA-ES parameters
        cc = (4 + n / (4 + n)) / ((n + 1) / 3 + 4 + n / (4 + n))
        cs = (n + 1.3) / (n + 3 + 2 * n / (n + 4))
        cmu = min(1 - cc - cs, 2 * (2 * n - 1) / (n + 2) ** 2)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        cmu_sigma = 1 + 2 * max(0, np.sqrt(cmu * (1 - cmu) / (cmu + 2)) - 1) + cmu

        def cma_es(x0, sigma, remaining_budget, rng, deadline):
            """Run CMA-ES from given starting point and sigma."""
            nonlocal call_count, best_x, best_f
            
            # State variables
            mean = x0.copy().astype(float)
            C = np.eye(n)  # Covariance matrix
            C_sqrt_inv = np.eye(n)  # Inverse square root of C
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Track best
            local_best_x = mean.copy()
            local_best_f = float('inf')
            
            evals_used = 0
            
            # Main loop
            while evals_used < remaining_budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                w = np.ones(lambda_pop)
                w[:lambda_pop//2] = 2.0  # Weight top half more
                w = w / w.sum()
                mu = lambda_pop // 2  # Number of selected individuals
                
                # Sample population
                Z = rng.standard_normal((lambda_pop, n))
                # Use C_sqrt_inv for sampling (approximate inverse sqrt)
                # For simplicity, use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                    samples = mean + sigma * (Z @ L.T)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    samples = mean + sigma * Z
                
                # Evaluate
                F = np.array([f(x) for x in samples])
                evals_used += lambda_pop
                
                if time.time() > deadline:
                    break
                
                # Sort by fitness
                idx = np.argsort(F)
                Y = samples[idx]
                F_sorted = F[idx]
                
                # Update best
                if F_sorted[0] < local_best_f:
                    local_best_f = F_sorted[0]
                    local_best_x = Y[0].copy()
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                
                # Weighted mean of selected
                y_w = np.sum(w[:mu, None] * Y[:mu], axis=0)
                
                # Update step size
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ (y_w / sigma)
                sigma *= np.exp((ds / ds) * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update evolution path
                mu_eff = 1.0 / np.sum(w[:mu] ** 2)
                p_c = (1 - cc) * p_c + cc * np.sqrt(mu_eff * (2 - cc) * cc) * C_sqrt_inv @ y_w
                
                # Update covariance matrix
                delta_heavy = 1 if np.linalg.norm(p_c) ** 2 / (1 - (1 - cc) ** (2 * (evals_used + 1) // lambda_pop)) > (1 + 2/cc) * (n + 1) else 0
                rank_mu = np.outer(y_w, y_w)
                rank_one = np.outer(p_c, p_c)
                C = (1 + cmu * (mu_eff - 2) - cc) * C + cmu * rank_mu + (1 + 2 * delta_heavy * cc * (1 - cc) - cc) * rank_one
                
                # Re-estimate C_sqrt_inv
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(n)
                
                # Update mean
                mean = mean + sigma * y_w
                
                # Convergence check
                if sigma < 1e-10 or np.all(np.abs(C - np.eye(n)) < 1e-12):
                    break
                
                # Stagnation check: if no improvement for many generations
                if evals_used > 50 and local_best_f == best_f and sigma < 0.01:
                    break
            
            return local_best_x, local_best_f, evals_used

        # Multi-start CMA-ES
        num_starts = 3
        budget_per_start = budget // num_starts
        
        for i in range(num_starts):
            if time.time() > deadline or call_count >= budget:
                break
            
            if i == 0:
                # First start at origin
                x0 = np.zeros(d)
                sigma = 2.0  # Large initial step
            else:
                # Subsequent starts at random points
                x0 = np.random.default_rng(42 + i).uniform(-5, 5, d)
                sigma = 1.0
            
            f0 = f(x0)
            call_count += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            remaining = budget - call_count
            if remaining < 2 * lambda_pop:
                break
            
            x_opt, f_opt, used = cma_es(x0, sigma, remaining, np.random.default_rng(1000 + i), deadline)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Final polish: run CMA-ES from best point with small sigma
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining >= 2 * lambda_pop:
                x_opt, f_opt, used = cma_es(best_x, 0.1, remaining, np.random.default_rng(2000), deadline)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
