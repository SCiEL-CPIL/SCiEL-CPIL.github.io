"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for Katsuura (f23) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if check_deadline():
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Multi-start strategy: several CMA-ES runs with different initializations
        # Katsuura is highly multimodal, so we spread starts across the box
        n_starts = 5
        # Allocate budget: first start gets more, others get less
        # Start 1: 200 calls, Starts 2-5: 70 calls each
        start_budgets = [200, 70, 70, 70, 70]
        
        # Initial points: spread across the box
        # Use a Latin hypercube-like spread plus some random
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Corners and midpoints
        for i in range(1, n_starts):
            pt = rng.uniform(-4, 4, dim)
            initial_points.append(pt)

        for start_idx in range(n_starts):
            if calls_made >= budget or check_deadline():
                break
            
            x0 = initial_points[start_idx]
            # Evaluate initial point
            f0 = evaluate(x0)
            if np.isinf(f0):
                break
            
            # CMA-ES parameters
            # Initial sigma: large enough to explore
            sigma = 3.0 / dim  # ~0.6 for dim=5
            # Population size
            lambda_ = 4 + int(3 * np.log(dim))  # ~12 for dim=5
            # Weights for recombination
            mu = lambda_ // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            # Step size adaptation
            c_sigma = (mueff + 2) / ((dim + 2) ** 2 + mueff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            # Covariance matrix adaptation
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            # Initialize
            x_mean = x0.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            # Precompute inverse square root of C
            evals, eigvecs = np.linalg.eigh(C)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-10))) @ eigvecs.T
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
            
            # Number of iterations for this start
            max_iter = start_budgets[start_idx]
            iter_count = 0
            
            while iter_count < max_iter:
                if calls_made + lambda_ > budget or check_deadline():
                    # Evaluate only as many as we can
                    lambda_this = max(1, min(lambda_, budget - calls_made))
                else:
                    lambda_this = lambda_
                
                # Generate offspring
                ys = []
                fs = []
                for k in range(lambda_this):
                    if check_deadline():
                        break
                    # Sample
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if np.isinf(fy):
                        break
                    ys.append(y)
                    fs.append(fy)
                
                if len(ys) < 2:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update mean
                y_w = np.array(ys[:mu])
                x_new_mean = np.sum(weights[:, None] * y_w, axis=0)
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
                h_sig = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n else 0.0
                p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma
                
                # Update covariance
                delta_h = (h_sig - 1) * c_c * (2 - c_c)
                C = (1 - c_c * (c_1 + c_mu) + delta_h) * C + \
                    c_1 * (np.outer(p_c, p_c) - C) + \
                    c_mu * np.sum(weights[:, None, None] * np.outer(y_w - x_mean, y_w - x_mean), axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update C_sqrt using Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_sqrt).T
                except np.linalg.LinAlgError:
                    # Fall back to eigenvalue decomposition
                    evals, eigvecs = np.linalg.eigh(C)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(evals, 1e-10)))
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-10)))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                x_mean = x_new_mean
                iter_count += 1
                
                if check_deadline():
                    break

        # Return the best solution found
        if best_x is None:
            # Should not happen, but safety
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
