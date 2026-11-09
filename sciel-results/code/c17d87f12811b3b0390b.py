# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer.
    
    Strategy:
    1. Global Initialization: Sample a diverse population of points to locate the basin.
       We use a mix of Latin Hypercube Sampling and random points to ensure coverage.
       The best point from this phase seeds the CMA-ES.
    2. CMA-ES Optimization: Run CMA-ES to converge to the minimum.
       CMA-ES adapts the covariance matrix of the search distribution, allowing it to
       align with the sharp ridge of f13 much more effectively than Nelder-Mead, which
       can struggle with ill-conditioned or sharp geometries if the simplex deforms poorly.
       CMA-ES is robust, derivative-free, and well-suited for low-dimensional continuous
       optimization with limited budget.
    3. Budget Management: Strictly monitor function calls and wall-clock time.
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
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Phase 1: Initialization
        # Sample 10 points using LHS to cover the space
        n_init = 10
        perm_matrix = np.zeros((dim, n_init), dtype=int)
        for d in range(dim):
            perm_matrix[d, :] = rng.permutation(n_init)
        
        init_points = []
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                bin_idx = perm_matrix[d, i]
                low = -5.0 + bin_idx * (10.0 / n_init)
                high = -5.0 + (bin_idx + 1) * (10.0 / n_init)
                x[d] = rng.uniform(low, high)
            init_points.append(x)
        
        # Evaluate initial points
        for x in init_points:
            fx = timed_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Step size sigma
        sigma = 1.0
        # Mean vector
        mean = best_x.copy()
        # Covariance matrix C
        C = np.eye(dim)
        # Inverse covariance matrix
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = lam // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Cholesky decomposition of C
        try:
            B_C = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B_C = np.eye(dim)
        
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
            
            # Generate population
            Y = rng.standard_normal((lam, dim))
            X = mean + sigma * (Y @ B_C.T)
            
            # Evaluate population
            F = []
            X_valid = []
            for i in range(lam):
                fx = timed_call(X[i])
                if fx is None:
                    break
                F.append(fx)
                X_valid.append(X[i])
            
            if len(F) < mu:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = np.array(X_valid)[order]
            F_sorted = np.array(F)[order]
            
            # Update mean
            mean_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution path p_sigma
            y_sigma = (mean_new - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Hessian norm of p_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) <
                       (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            
            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update covariance matrix
            # Delta h term
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-one update
            C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            z = np.linalg.solve(B_C.T, X_sorted[:mu].T - mean.T)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(z, z), axis=2)
            C += c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Add small regularization if needed
            try:
                B_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B_C = np.linalg.cholesky(C)
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            mean = mean_new
            
            # Convergence check
            if np.linalg.norm(mean - best_x) < 1e-6 and np.all(np.abs(F_sorted[:mu] - F_sorted[0]) < 1e-8):
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
