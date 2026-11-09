"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        # Wrap func to count calls and enforce budget
        call_count = 0
        budget = self.budget
        dim = self.dim
        
        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                raise Exception("Budget exceeded")
            call_count += 1
            return func(x)

        best_x = None
        best_f = float('inf')
        
        # Multi-start strategy
        # We divide the budget among several runs.
        # CMA-ES is population-based, so it consumes budget faster per "step" but converges faster.
        # A typical CMA-ES run for dim=5 might need ~100-200 evaluations to converge well.
        # With 500 budget, we can afford 2-3 restarts if we are smart, or just one long run.
        # Given Rosenbrock's narrow valley, a single well-tuned CMA-ES often outperforms restarts 
        # unless the initial point is very bad. However, random initialization can be far.
        # Let's try 3 restarts with budget 150 each, or dynamic restarts.
        
        # Strategy: Run CMA-ES. If it stagnates (no improvement for many steps) or time runs out, restart.
        # Actually, standard CMA-ES handles stagnation via sigma adaptation and restarts in the algorithm itself.
        # But for a black-box with limited budget, explicit multi-start is safer against bad initializations.
        
        num_restarts = 3
        budget_per_restart = budget // num_restarts
        
        rng = np.random.default_rng(42)
        
        for r in range(num_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline - 10: # Leave 10s for finishing up
                break
                
            # Initialize
            # Start near center or random? Random is safer for global search, but Rosenbrock valley is specific.
            # Let's start at origin (0,0,0,0,0) for the first restart, then random for others?
            # The minimizer is in [-4,4]^5. Origin is a decent guess.
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)
                
            # Initial sigma
            sigma = 1.0
            
            # CMA-ES parameters
            # lambda = 4 + floor(3 ln n)
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            
            # Initialize matrices
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Evaluate initial point
            f_x = counted_func(x)
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            # Main loop for this restart
            while call_count + lam <= budget:
                if time.time() > deadline - 5:
                    break
                    
                # Generate population
                Y = np.array([x + sigma * C_sqrt @ rng.standard_normal(dim) for _ in range(lam)])
                
                # Evaluate population
                F = np.array([counted_func(y) for y in Y])
                
                # Update best found
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = Y[min_idx].copy()
                    
                # Sort
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                
                # Weighted mean
                x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
                
                # Update step size control variable
                # Need inverse sqrt(C)
                # Since C is symmetric positive definite, we can use eigh
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # h_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                norm_C_sqrt_inv = np.linalg.norm(C_inv_sqrt @ p_sigma) # This is not quite right for h_sigma calculation in standard CMA
                # Standard h_sigma:
                # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-2*step)/n_steps) < (1.4 + 1/dim) * E[||N(0,I)||]
                # Simplified:
                expected_norm = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
                h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - 2*sigma**2)) < (1.4 + 1/dim) * expected_norm else 0.0 # Approximation
                
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
                
                # Update C
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                # Rank-one update
                y_sigma = (x_new - x) / sigma
                C = (1 + c_1 * (c_mu + h_sigma * (1 - h_sigma)) - c_1) * C + c_1 * np.outer(y_sigma, y_sigma)
                
                # Rank-mu update
                C += c_mu * np.sum(weights[:mu][:, None, None] * np.outer(Y[:mu] - x, Y[:mu] - x) / sigma**2, axis=0)
                
                # Update C_sqrt via eigen decomposition (expensive but robust for small dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / expected_norm - 1))
                
                # Check termination
                if np.linalg.norm(x_new - x) < 1e-6 * max(1.0, np.linalg.norm(x)):
                    break
                if sigma < 1e-8:
                    break
                    
                x = x_new
                
        return best_x, best_f
# EVOLVE-BLOCK-END
