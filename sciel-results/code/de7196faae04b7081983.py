"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = float('inf')
        deadline = time.time() + 285

        def count_eval(x):
            nonlocal calls_made
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Coarse random sampling to find promising basins
        # Rastrigin is multimodal; we need to sample widely first.
        n_coarse = 20
        coarse_x = []
        coarse_f = []
        
        for _ in range(n_coarse):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = count_eval(x)
            coarse_x.append(x)
            coarse_f.append(fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if not coarse_x:
            # Fallback if even coarse search fails
            x = rng.uniform(-5, 5, self.dim)
            fx = count_eval(x)
            return x, fx

        # Sort coarse results
        order = np.argsort(coarse_f)
        
        # Phase 2: Run CMA-ES from the best few candidates
        # We pick the top 3 candidates to start CMA-ES
        n_starts = min(3, len(coarse_x))
        top_indices = order[:n_starts]
        
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Allocate budget to each CMA-ES run
        # CMA-ES in 5D typically needs ~100-300 evals for good convergence
        budget_per_cma = max(50, remaining_budget // n_starts)

        for idx in top_indices:
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            x0 = coarse_x[idx].copy()
            f0 = coarse_f[idx]
            
            # Initialize CMA-ES parameters
            # Sigma (step size) should be moderate to explore the local basin
            # but not too large to jump out.
            sigma = 0.5 
            # Population size
            # Standard rule: lambda = 4 + floor(3 * log(N))
            # For N=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8
            # Let's use 10 for slightly more robustness
            lam = 10
            
            # CMA-ES state
            # Mean
            mean = x0.copy()
            # Covariance matrix (initially identity)
            C = np.eye(self.dim)
            # Eigen decomposition for updating C
            # We will use a simplified update or standard CMA-ES update
            # For simplicity and speed, we use the standard update rule.
            
            # Weights for selection
            # mu = lambda / 2
            mu = lam // 2
            # Weights sum to 1
            w = np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            # Variance effective number
            mueff = 1.0 / np.sum(w**2)
            
            # Learning rates
            cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
            cs = (mueff + 2) / (self.dim + mueff + 5)
            c1 = 2 / ((self.dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
            
            # Path variables
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Initial step size
            # Start with a slightly larger sigma to escape shallow local minima
            # but reduce it quickly if stuck.
            # Actually, standard CMA-ES starts with a given sigma.
            # Let's start with 0.5.
            
            # Main loop for this CMA-ES instance
            max_iter = budget_per_cma
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Generate population
                # Sample from N(mean, sigma^2 * C)
                # We need to sample from C. Use Cholesky or eigen.
                # For 5D, eigen is cheap.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive definiteness
                eigvals = np.maximum(eigvals, 1e-12)
                # Sample
                z = rng.standard_normal((lam, self.dim))
                # Transform: eigvecs * diag(sigma * sqrt(eigvals)) * z
                # Actually, N(0, sigma^2 C) = eigvecs @ (sigma * sqrt(eigvals) * z.T).T
                samples = mean + (sigma * np.sqrt(eigvals)) * (z @ eigvecs.T)
                
                # Evaluate population
                fvals = []
                for i in range(lam):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    xi = np.clip(samples[i], -5, 5)
                    fi = count_eval(xi)
                    fvals.append(fi)
                    
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if len(fvals) < mu:
                    # Not enough evaluations, break
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(fvals)
                # Take top mu
                selected_idx = sorted_idx[:mu]
                x_selected = samples[selected_idx]
                f_selected = np.array(fvals)[selected_idx]
                
                # Update mean
                mean_new = np.sum(w[:, None] * x_selected, axis=0)
                
                # Update path variables
                # We need the transformation matrix B and D
                # B = eigvecs, D = sqrt(eigvals)
                # Transformation: (mean_new - mean) / (sigma * D)
                # Note: C = B @ diag(D^2) @ B.T
                # So C^(-1/2) = B @ diag(1/D) @ B.T
                # The path update uses C^(-1/2)
                
                # Calculate C^(-1/2) (approx)
                # Since we have eig, C^(-1/2) = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - mean) / sigma
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (max_iter + 1))) / np.sqrt(1 - (1 - cs)**(2 * (max_iter + 1))) # Approximation, usually just check against threshold
                # Standard threshold: c_chi = sqrt(self.dim) * (1 - (1 - cs)**2)
                c_chi = np.sqrt(self.dim) * (1 - (1 - cs)**2)
                if np.linalg.norm(p_sigma) < (1.4 + 2 / (self.dim + 1)) * c_chi:
                    h_sigma = 1
                else:
                    h_sigma = 0
                
                p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - mean) / sigma
                
                # Update covariance
                # C = (1 + mu_eff * (cmu + c1) - c1) * C
                #    + cmu * sum(wi * (xi - mean_old) @ (xi - mean_old).T)
                #    + c1 * p_c @ p_c.T
                
                # Compute the rank-mu update
                diff = x_selected - mean
                rank_mu_update = np.sum((w[:, None] * diff) @ diff.T, axis=0)
                
                # Update C
                C = (1 + mu * (cmu + c1) - c1) * C + cmu * rank_mu_update + c1 * np.outer(p_c, p_c)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update mean
                mean = mean_new
                
                # Adapt sigma
                # Standard CMA-ES sigma update
                # sigma = sigma * exp(cs / dsigma * (||p_sigma|| / c_chi - 1))
                dsigma = 1 + 2 * max(0, (mueff - 1) / (self.dim + 1)) + cs
                sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(p_sigma) / c_chi - 1))
                
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)
                
                # Check convergence
                # If sigma is too small or step size is too small
                if sigma < 1e-4:
                    break
                # If the spread of the population is very small
                # (Approximated by checking if eigenvalues are small)
                if np.max(eigvals) * sigma**2 < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
