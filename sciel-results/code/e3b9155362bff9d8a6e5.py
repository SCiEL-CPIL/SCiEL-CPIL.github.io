"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Multi-Start strategy for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, time_limit):
        n = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        
        # Population size
        # For small budgets, keep lambda small
        lambda_ = int(4 + 3 * np.log(n))
        lambda_ = min(lambda_, budget)
        
        # Weights
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n**1.6) + c_sigma) / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1.0 / mu_eff)) / ((n + 2)**2 + mu_eff))
        
        # Matrices
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Initial step size adaptation
        # If x0 is random, we might want a larger initial sigma
        # But standard CMA-ES starts with sigma0
        
        for i in range(budget):
            if time.time() > time_limit:
                break
                
            # Generate population
            # Update C inverse sqrt approximation (simplified for speed, using Cholesky)
            # For n=5, full matrix ops are cheap
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
                C = np.eye(n)
                
            # Sample
            ys = np.array([sigma * rng.multivariate_normal(np.zeros(n), C) for _ in range(lambda_)])
            # Actually, multivariate_normal with covariance C is slow. 
            # Better: z ~ N(0, I), y = sigma * L @ z
            zs = rng.standard_normal((lambda_, n))
            ys = sigma * (zs @ L.T)
            
            ps = x + ys
            # Clip to bounds
            ps = np.clip(ps, -5, 5)
            
            # Evaluate
            fs = np.array([func(p) for p in ps])
            
            # Sort
            o = np.argsort(fs)
            ps = ps[o]
            fs = fs[o]
            
            # Update mean
            w_sum = w.sum()
            new_x = np.sum(w[:, np.newaxis] * ps[:mu], axis=0)
            
            # Update step size
            y_mean = (new_x - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            h_sigma = 1 if (np.linalg.norm(p_sigma)**2 / ((1 - (1 - c_sigma)**(2*(i+1))) * n) < 
                           (2 / c_sigma + 4) * (1 - (1 - c_sigma)**(2*(i+1))) / n) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Rank-one update
            C = (1 + c_1 * c_mu - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            ys_mu = (ps[:mu] - x) / sigma
            C = (1 + c_1 * c_mu - c_1 + c_mu) * C - c_mu * np.sum(w[:, np.newaxis, np.newaxis] * 
                  np.einsum('i,jk->ijk', ys_mu, ys_mu), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x = new_x
            fx = fs[0]
            
            # Accept if better
            if fs[0] < fx:
                 # Already set fx = fs[0] above, so this is redundant if we track best separately
                 pass

        return x, fx

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 280s
        deadline = start_time + 280
        
        rng = np.random.default_rng(42)
        
        # Multi-Start Strategy
        # Split budget into exploration and exploitation
        # e.g., 5 starts of 50 evals, then 250 evals for best
        n_starts = 5
        evals_per_start = 50
        total_explore = n_starts * evals_per_start
        
        if self.budget <= 100:
            # Just run one long CMA-ES
            x0 = rng.uniform(-5, 5, self.dim)
            return self._cma_es(func, x0, 2.0, self.budget, rng, deadline)
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Exploration
        for i in range(n_starts):
            if time.time() > deadline:
                break
            # Random start, biased towards center slightly? 
            # Lunacek has a global minimum near origin and a local minimum far away.
            # Uniform in [-5, 5] is safe.
            x0 = rng.uniform(-5, 5, self.dim)
            # Start with moderate sigma
            x, f = self._cma_es(func, x0, 1.5, evals_per_start, rng, deadline)
            
            if f < best_f:
                best_f = f
                best_x = x
        
        # Phase 2: Exploitation
        remaining_budget = self.budget - total_explore
        if remaining_budget > 0 and best_x is not None:
            if time.time() < deadline:
                # Refine from best point found
                # Use a smaller sigma or let CMA-ES adapt
                x, f = self._cma_es(func, best_x, 0.5, remaining_budget, rng, deadline)
                if f < best_f:
                    best_f = f
                    best_x = x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
