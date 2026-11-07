"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with tuned population size for low dimensions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin before 300s limit
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Using lambda=8 for dim=5 to allow for ~62 generations (500/8)
        # This increases the frequency of adaptation steps compared to lambda=10.
        lambda_pop = 8
        mu = lambda_pop // 2 # Selection pressure
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # State variables
        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 2.0
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main loop
        while call_count < budget and time.time() < deadline:
            # Generate population
            Z_list = []
            X = []
            F = []
            
            for i in range(lambda_pop):
                # Sample Z
                Z = rng.standard_normal(dim)
                # Transform Z using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ Z
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    y = Z
                
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                
                Z_list.append(Z)
                X.append(x)
                F.append(f)
                
                if call_count >= budget or time.time() > deadline:
                    break
            
            if len(X) < mu:
                break
                
            # Sort by fitness
            sort_idx = np.argsort(F)
            X = [X[i] for i in sort_idx]
            Z_list = [Z_list[i] for i in sort_idx]
            F = [F[i] for i in sort_idx]
            
            # Update Mean
            old_mean = mean.copy()
            mean = np.sum([w * X[i] for i, w in enumerate(weights)], axis=0)
            
            # Update Covariance Matrix
            # Normalize Z for covariance update
            Z_norm = [Z_list[i] / sigma for i in range(mu)]
            
            # Rank-mu update
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(Z_norm[i], Z_norm[i])
            
            # Update pc and ps
            # Note: Using (mean - old_mean) / sigma is the standard normalized step
            step = (mean - old_mean) / sigma
            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * step
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * step
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_update
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Adapt sigma
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Clamp sigma
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        return best_x, best_f

# EVOLVE-BLOCK-END
