"""Rewritten optimizer: grid-search warm-up + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid grid-search warm-up + CMA-ES refinement for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Deterministic grid-search warm-up ---
        # Use a small number of calls to bracket the minimum.
        # We'll use a coarse grid with a few points per dimension.
        # For dim=5, a 3^5 = 243 point grid is too many for a small budget.
        # Instead, use a hierarchical approach: start with a very coarse grid.
        
        # Let's use a smarter approach: 
        # 1. First, evaluate a few random points to get a sense of scale.
        # 2. Then do a coarse grid search.
        
        # Actually, let's use a deterministic space-filling approach.
        # For 5D, we can use a Halton sequence for quasi-random sampling.
        
        # Phase 1: Quasi-random warm-up with ~100 calls
        n_warmup = min(100, budget // 5)
        
        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    # Prime number for dimension j
                    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][j]
                    f = 1.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= prime
                        i_val = i_val // prime
                    points[i, j] = f
            return points
        
        # Generate Halton points and map to [-5, 5]^dim
        halton_pts = halton_sequence(n_warmup, dim)
        halton_pts = halton_pts * 10 - 5  # Map [0,1] to [-5,5]
        
        # Evaluate all warm-up points
        best_x = None
        best_f = np.inf
        for i in range(n_warmup):
            x = np.clip(halton_pts[i], -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        calls_made = n_warmup
        
        # --- Phase 2: CMA-ES refinement ---
        # Initialize CMA-ES around the best point found
        # Use a reasonable initial sigma based on the grid spacing
        
        # Estimate the local scale from the grid
        # The grid has spacing of about 10/n_warmup^(1/dim) in each dimension
        # For n_warmup=100, dim=5: spacing ~ 10/100^0.2 ~ 10/2.51 ~ 4
        # So use sigma ~ 2.0 initially
        
        sigma = 2.0
        x = best_x.copy()
        fx = best_f
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim))
        # For dim=5: lam = 4 + floor(3*1.609) = 4 + 4 = 8
        
        # Covariance matrix C
        C = np.eye(dim)
        # Eigenvalue decomposition for sampling
        # We'll use a simplified CMA-ES
        
        # Step size control
        cs = (lam / 2 + 1) ** (-1.0 / (dim + 17))  # Path length adaptation for sigma
        # Actually, let's use standard CMA-ES parameters
        
        # Standard CMA-ES
        # Population size
        # lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim))
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial mean
        mean = x.copy()
        # Initial sigma
        sigma = 1.5
        
        # Budget for CMA-ES
        remaining = budget - calls_made
        if remaining <= 0:
            return x, fx
        
        # Run CMA-ES
        for gen in range(remaining // lam):
            if calls_made >= budget:
                break
            
            # Generate offspring
            # Compute eigen decomposition of C for sampling
            evals, evecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            evals = np.maximum(evals, 1e-20)
            B = evecs
            D = np.sqrt(evals)
            
            # Sample from N(0, C)
            z = rng.standard_normal((lam, dim))
            y = z @ (B * D).T  # y = z @ B * D
            
            # Update path lengths
            y_sigma = y / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (weights[:mu] @ y_sigma[:mu].T)
            
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) <
                       (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
            h_sigma = float(h_sigma)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                  (weights[:mu] @ y[:mu].T)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_c = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])
            
            C = (1 + c_1 * (c_mu - c_1) - c_1) * C + c_1 * (rank_c + delta_h * rank_c) + \
                c_mu * (rank_mu - C)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-10 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) /
                                                  np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) - 1))
            
            # Generate candidates
            candidates = []
            for i in range(lam):
                xi = mean + sigma * y[i]
                xi = np.clip(xi, -5, 5)
                candidates.append(xi)
            
            # Evaluate candidates
            fvals = []
            for xi in candidates:
                if calls_made >= budget:
                    break
                fi = func(xi)
                fvals.append(fi)
                calls_made += 1
            
            if len(fvals) == 0:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = y[idx]
            fvals_sorted = np.array(fvals)[idx]
            
            # Update mean
            mean = mean + sigma * (weights[:mu] @ y_sorted[:mu].T)
            mean = np.clip(mean, -5, 5)
            
            # Update best solution
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = candidates[idx[0]].copy()
            
            # Stopping criteria
            if sigma < 1e-10:
                break
            if gen > 0 and np.all(np.abs(y_sorted[:mu]) < 1e-10):
                break
        
        return x, fx
# EVOLVE-BLOCK-END
