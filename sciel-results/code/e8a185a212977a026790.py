# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES variant with adaptive population size.
    
    Strategy:
    1. Initialize with a diverse population of random points.
    2. Run CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with:
       - Adaptive population size that shrinks as convergence progresses.
       - Covariance matrix adaptation to handle anisotropic landscapes (like Sharp Ridge).
       - Step size adaptation via CSA (Cumulative Step Adaptation) or simple rule.
    3. This approach is more robust than (1+1)-ES for navigating ridges and
       can converge faster in high dimensions with structured landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Start with a population of size lambda = 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + 3 = 7. Let's use a slightly larger initial population.
        n_init = 20
        candidates = []
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = timed_call(x)
            if fx is not None:
                candidates.append((fx, x))
        
        if not candidates:
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Sort and pick the best as initial mean
        candidates.sort(key=lambda item: item[0])
        mean = candidates[0][1].copy()
        best_f = candidates[0][0]
        best_x = candidates[0][1].copy()

        # Phase 2: CMA-ES
        # Parameters
        dim = self.dim
        # Initial population size
        lam = 4 + int(3 * np.log(dim))
        # For dim=5, lam=7. Let's start with a larger population for better exploration.
        lam = max(10, 2 * dim)  # 10 for dim=5
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Eigenvalues and eigenvectors for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Learning rates
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track progress for adaptive population size
        prev_best_f = best_f
        stagnation_count = 0
        min_lam = 4
        
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
            
            # Adaptive population size: shrink if no progress
            if best_f < prev_best_f - 1e-8:
                prev_best_f = best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Shrink population if stagnating
            if stagnation_count > 50 and lam > min_lam:
                lam = max(min_lam, lam - 2)
            
            # Sample population
            pop = []
            for i in range(lam):
                # Sample z ~ N(0, 1)
                z = rng.standard_normal(dim)
                # Project to eigenvectors
                z_eig = eigvecs.T @ z
                # Scale by sqrt of eigenvalues
                z_scaled = z_eig * np.sqrt(eigvals)
                # Project back
                z = eigvecs @ z_scaled
                # Generate point
                x = mean + sigma * z
                x = np.clip(x, -5, 5)
                fx = timed_call(x)
                if fx is None:
                    break
                pop.append((fx, x))
            
            if not pop:
                break
            
            # Sort population
            pop.sort(key=lambda item: item[0])
            
            # Update best
            if pop[0][0] < best_f:
                best_f = pop[0][0]
                best_x = pop[0][1].copy()
            
            # Select top mu individuals
            selected = [x for _, x in pop[:mu]]
            
            # Weighted mean
            new_mean = np.sum(weights[:, None] * np.array(selected), axis=0)
            
            # Update evolution paths
            old_mean = mean.copy()
            mean = new_mean
            
            # p_sigma update
            z_mean = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1)) / np.sqrt(eigvals.sum())) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1))
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean
            
            # Update covariance matrix
            delta_h = (h_sigma - 1) * c_c * (2 - c_c) * p_c
            C = (1 + c_1 * c_mu + c_1 * c_c * (2 - c_c) - c_1 - c_mu) * C + \
                c_1 * (p_c @ p_c.T + delta_h * p_c + p_c * delta_h) + \
                c_mu * np.sum(weights[:, None, None] * np.outer(np.array(selected) - old_mean, np.array(selected) - old_mean), axis=0)
            
            # Re-orthogonalize covariance matrix to ensure positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1)) - np.sqrt(1 - (1 - c_sigma) ** 2) * np.sqrt(dim)))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Check for convergence
            if np.all(eigvals < 1e-10):
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
