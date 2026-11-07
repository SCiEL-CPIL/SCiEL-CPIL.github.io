"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin for 330s limit
        rng = np.random.default_rng(0)
        
        # Weierstrass f16 is multimodal with many local minima.
        # A population-based approach with CMA-ES features is more robust.
        
        dim = self.dim
        budget = self.budget
        
        # Initial population
        pop_size = min(20, max(10, budget // 25))
        
        # Use a mix of global and local search
        # Start with a broad exploration
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Broad global search with random restarts and local refinement
        # We'll use a simple evolutionary strategy with adaptive sigma
        
        # Initial point
        x = rng.uniform(-4, 4, dim)  # Minimizer is in [-4, 4]^5
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.0
        success_ratio = 0.5  # Target success rate for step size adaptation
        
        # Adaptive step size controller
        # Use a simple version of 1/5 rule or similar
        
        # Population based CMA-ES lite
        # We'll maintain a population and adapt the distribution
        
        # Initialize population
        pop = np.array([x] + [rng.uniform(-5, 5, dim) for _ in range(pop_size - 1)])
        fvals = np.array([fx] + [func(p) for p in pop[1:]])
        
        # Track best
        idx_best = np.argmin(fvals)
        best_x = pop[idx_best].copy()
        best_f = fvals[idx_best]
        
        # CMA-ES parameters
        lambda_ = pop_size
        mu = max(1, lambda_ // 2)
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 1.0
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Memory
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        d_norm = np.sqrt(dim)
        
        # Track progress
        no_improve = 0
        restart_threshold = 50  # Restart if no improvement for this many iterations
        
        for it in range(1, budget):
            if time.time() > deadline:
                break
            
            # Check for stagnation
            if no_improve > restart_threshold and it > 50:
                # Try a restart from a new random point
                x_new = rng.uniform(-4, 4, dim)
                fx_new = func(x_new)
                if fx_new < best_f:
                    best_x = x_new.copy()
                    best_f = fx_new
                    no_improve = 0
                    # Reset some CMA-ES state
                    sigma = 1.0
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    C = np.eye(dim)
                else:
                    no_improve += 1
                continue
            
            # Generate population
            # Use CMA-ES sampling
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim)
            
            # Sample
            z = rng.standard_normal((lambda_, dim))
            pop_new = np.array([best_x + sigma * (L @ z[i].T).T for i in range(lambda_)])
            # Clip to bounds
            pop_new = np.clip(pop_new, -5, 5)
            
            # Evaluate
            fvals_new = np.array([func(p) for p in pop_new])
            
            # Sort by fitness
            order = np.argsort(fvals_new)
            pop_new = pop_new[order]
            fvals_new = fvals_new[order]
            
            # Update best
            if fvals_new[0] < best_f:
                best_f = fvals_new[0]
                best_x = pop_new[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean
            y_w = np.sum(weights[:, np.newaxis] * pop_new[:mu], axis=0)
            y_w = y_w / sigma
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance matrix
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)
            h_sigma = 1 if h_sigma else 0
            
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (
                (1 + delta_h - c_sigma) * np.outer(p_sigma, p_sigma) - np.eye(dim)
            )
            C = (C + C.T) / 2  # Ensure symmetry
            
            # Update step size
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 0.01, 10)
            
            # Ensure C is positive definite
            try:
                np.linalg.cholesky(C)
            except:
                # Add small regularization
                C = C + 1e-4 * np.eye(dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
