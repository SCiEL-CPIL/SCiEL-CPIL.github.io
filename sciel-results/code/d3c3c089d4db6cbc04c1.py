"""Rewritten program using CMA-ES for better handling of non-separable functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Continuous optimizer using CMA-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # Safety margin

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Stage 1: Initial random sampling
        # Use a small number of points to estimate scale and find a good start
        n_init = min(10, budget)
        init_xs = []
        init_fs = []
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = safe_call(x)
            init_xs.append(x)
            init_fs.append(fx)
        
        if calls >= budget:
            return best_x, best_f

        # Sort by fitness to find the best starting point
        sorted_indices = np.argsort(init_fs)
        best_idx = sorted_indices[0]
        x_mean = init_xs[best_idx].copy()
        
        # Estimate initial sigma based on spread of good solutions or default
        # If we have at least 2 points, use distance to second best
        if len(init_xs) > 1:
            second_best_idx = sorted_indices[1]
            dist = np.linalg.norm(init_xs[best_idx] - init_xs[second_best_idx])
            sigma = max(dist / 2.0, 0.1)
            sigma = min(sigma, 2.0) # Cap initial sigma
        else:
            sigma = 1.0

        # Stage 2: CMA-ES
        # Parameters for CMA-ES
        lambda_ = 10  # Population size
        # Ensure we don't exceed budget
        if calls + lambda_ > budget:
            lambda_ = max(1, budget - calls)

        # Weights
        mu = lambda_ // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / ((dim + 1) + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # CMA-ES State
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # chi_n: Expected value of the norm of a standard normal vector
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        restarts = 0
        max_restarts = 3
        no_improve_count = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Check for stagnation or convergence
            # Simple heuristic: if sigma is very small or no improvement for many generations
            if sigma < 1e-6:
                break

            # Generate population
            xs = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                fx = safe_call(x)
                xs.append((x, fx))

            if len(xs) < 2:
                break

            # Sort by fitness
            xs.sort(key=lambda t: t[1])
            
            # Update best
            if xs[0][1] < best_f:
                best_f = xs[0][1]
                best_x = xs[0][0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If no improvement for a while, consider restart
            if no_improve_count > 20 and restarts < max_restarts:
                restarts += 1
                # Restart from a random point or near best
                if rng.random() < 0.5:
                    x_mean = rng.uniform(-5.0, 5.0, dim)
                else:
                    x_mean = best_x + 1.0 * rng.standard_normal(dim)
                    x_mean = np.clip(x_mean, -5.0, 5.0)
                
                sigma = 1.0
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                # Consume one call for the restart point? 
                # Actually, the restart point is just the new mean. 
                # We should evaluate it to be consistent? 
                # The prompt says "Every call is recorded". 
                # If we change the mean, we don't necessarily need to call it immediately 
                # unless we want to update best_x. 
                # However, to be safe and consistent with "calls made", 
                # let's evaluate the new center if we have budget.
                if calls < budget:
                     fx_center = safe_call(x_mean)
                     if fx_center < best_f:
                         best_f = fx_center
                         best_x = x_mean.copy()
                
                continue # Skip adaptation for this generation

            # Adaptation
            # Weighted mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * xs[i][0]
            
            # Update CMA parameters
            # p_sigma
            old_sigma = sigma
            y_sigma = (new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Sigma update (1/5 rule equivalent)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # p_c
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Covariance update
            delta_h = (2 - h_sigma) * c_c - 1
            rank_one = np.outer(p_c, p_c)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = xs[i][0] - x_mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu / sigma**2
            
            # Re-orthogonalize C to prevent numerical issues
            # Use Cholesky or Eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            x_mean = new_mean
            x_mean = np.clip(x_mean, -5.0, 5.0)

        # Final polish if budget remains
        if calls < budget and best_x is not None:
            # Simple local search
            step_size = 0.01
            for _ in range(min(5, budget - calls)):
                if time.time() > deadline:
                    break
                x = best_x + step_size * rng.standard_normal(dim)
                x = np.clip(x, -5.0, 5.0)
                safe_call(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
