import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cmaes_run(start_x, start_sigma, budget_limit):
            """Run CMA-ES from a starting point until budget or convergence."""
            nonlocal calls_made, best_x, best_f
            
            # CMA-ES parameters
            pop_size = max(4 + int(3 * np.log(n)), 8)  # Standard population size
            lambda_ = pop_size
            mu = lambda_ // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # Step size control
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            
            # Mean update
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            # Initialize
            mean = start_x.copy()
            sigma = start_sigma
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # Will compute properly
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Precompute Cholesky
            def get_C_sqrt_inv(C_mat):
                L = np.linalg.cholesky(C_mat + 1e-10 * np.eye(n))
                return L.T  # Upper triangular, so C = L @ L.T, and we need C^(-1/2)
            
            # We'll use a simpler approach: maintain C and use eigendecomposition periodically
            # For efficiency in 5D, we can use direct methods
            
            gen = 0
            while calls_made < budget_limit and time.time() < deadline:
                # Generate population
                trials = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    # Transform using Cholesky of C
                    # C = L @ L.T, so x = mean + sigma * L @ z
                    try:
                        L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                        x = mean + sigma * L.T @ z
                    except np.linalg.LinAlgError:
                        # Fallback if C is not positive definite
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-10)
                        C_sqrt_inv = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                        x = mean + sigma * C_sqrt_inv @ z
                    
                    x = np.clip(x, -5, 5)
                    trials.append(x)
                
                # Evaluate
                fits = np.array([evaluate(t) for t in trials])
                
                # Sort by fitness
                idx = np.argsort(fits)
                x_sorted = [trials[i] for i in idx]
                f_sorted = fits[idx]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                
                # Select parents
                x_parents = np.array(x_sorted[:mu])
                
                # Weighted mean
                new_mean = np.sum(weights[:, None] * x_parents, axis=0)
                
                # Update step-size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * sigma / np.sqrt(np.trace(C)) * (new_mean - mean) / sigma if sigma > 0 else (1 - c_sigma) * p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - mean) / sigma if sigma > 1e-10 else (1 - c_sigma) * p_sigma
                
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(mueff + 1 - mueff) else 0
                
                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma if sigma > 1e-10 else (1 - c_c) * p_c
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None] * np.outer(x_parents - mean, x_parents - mean), axis=0)
                C = (1 + c_sigma * c_sigma * (sigma / np.sqrt(np.trace(C)) if sigma > 0 else 1) * (delta_h - 1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                
                # Update mean and sigma
                mean = new_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
                sigma = np.clip(sigma, 1e-6, 10)
                
                gen += 1
                
                # Convergence check
                if sigma < 1e-5 or np.max(np.abs(mean)) > 50:
                    break
            
            return mean, sigma, C

        # Multi-start CMA-ES strategy
        # Phase 1: Initial random sampling to find promising regions
        n_init = min(20, total_budget // 10)
        init_points = []
        for i in range(n_init):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = evaluate(x)
            init_points.append((f, x))
        
        init_points.sort()
        
        # Phase 2: Run CMA-ES from top starting points
        n_starts = min(3, len(init_points))
        for s in range(n_starts):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            start_f, start_x = init_points[s]
            # Start with moderate sigma
            start_sigma = 2.0
            
            # Budget for this CMA-ES run
            remaining = total_budget - calls_made
            cma_budget = max(remaining // (n_starts - s), 30)
            
            cmaes_run(start_x, start_sigma, calls_made + cma_budget)
        
        # Phase 3: Local refinement around best found
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Simple local search with decreasing step size
            sigma_local = 0.5
            for _ in range(3):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                for _ in range(min(15, total_budget - calls_made)):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    x = best_x + sigma_local * rng.standard_normal(n)
                    x = np.clip(x, -5, 5)
                    f = evaluate(x)
                sigma_local *= 0.5
        
        # Fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f
