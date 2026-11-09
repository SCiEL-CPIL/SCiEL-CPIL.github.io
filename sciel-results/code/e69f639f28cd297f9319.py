"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP surrogate."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a small random sample
        n_init = 10
        X = rng.uniform(-5, 5, (n_init, dim))
        Y = np.array([func(x) for x in X])

        # GP hyperparameters
        # Length scales: Weierstrass has oscillations at multiple scales
        # Start with moderate length scales
        lengthscales = np.ones(dim) * 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def matern52(k, ls):
            """Matern 5/2 kernel."""
            # k: (n, n) distance matrix
            # ls: length scale (scalar or array)
            if len(ls.shape) > 0:
                # Anisotropic: scale distances per dimension
                # k is already squared distances? No, let's compute properly.
                # Actually, for simplicity and speed, let's use isotropic or simple anisotropic.
                # Let's stick to isotropic first for stability, or simple per-dim.
                # The standard Matern-5/2 is:
                # K(x, x') = signal_var * (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)
                # where r = ||x - x'|| / ls
                pass
            
            r = k / ls
            return signal_var * (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)

        def compute_kernel(X1, X2, ls):
            """Compute Matern-5/2 kernel between X1 and X2."""
            # X1: (n1, dim), X2: (n2, dim)
            # ls: scalar or (dim,)
            if np.isscalar(ls):
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist) / ls
            else:
                # Anisotropic
                # Scale each dimension by 1/ls
                scaled_X1 = X1 / ls
                scaled_X2 = X2 / ls
                diff = scaled_X1[:, np.newaxis, :] - scaled_X2[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                r = np.sqrt(sq_dist)
            
            return signal_var * (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)

        # Simple hyperparameter optimization using grid search or random search over a few candidates
        # For efficiency, we'll do a very light optimization or fixed reasonable values.
        # Given the 300s limit and 500 calls, we need to be fast.
        # Let's fix some reasonable hyperparameters to avoid heavy optimization overhead.
        # Weierstrass f16 has a characteristic frequency. 
        # Let's try a fixed length scale of 0.5 or 1.0 and see.
        # Actually, adaptive length scale might be better.
        
        # Let's implement a simple acquisition function maximization.
        # EI(x) = (mu - y_best - xi) * Phi(z) + sigma * phi(z)
        # where z = (mu - y_best - xi) / sigma
        
        # To maximize EI, we'll use a grid search in the box [-5, 5]^dim.
        # With dim=5, a grid of 10^5 = 100,000 points is feasible per iteration?
        # 500 iterations * 100,000 points * kernel eval... might be too slow.
        # Let's use a smaller grid, e.g., 5^5 = 3125 points, or use L-BFGS-B on the acquisition function.
        
        # L-BFGS-B is better. We need gradients? No, SciPy's minimize can use numerical gradients.
        # But we need to evaluate the GP mean and variance many times.
        
        # Let's use a simpler approach: random restarts of a local search on the acquisition function,
        # or just a coarse grid + local refinement.
        
        # Given time constraints, let's use a grid of 4^5 = 1024 points per iteration.
        # 500 * 1024 = 512,000 GP evaluations total.
        # Each GP eval requires solving a linear system of size n (growing up to 500).
        # This is O(n^3) per acquisition step if we don't cache.
        # We need to precompute K(X, X) and its inverse/Cholesky.
        
        # Let's structure the loop:
        # 1. Maintain X, Y.
        # 2. Compute K = compute_kernel(X, X, ls) + noise_var * I
        # 3. Cholesky decompose K.
        # 4. For each candidate x in grid:
        #    a. Compute k_x = compute_kernel(x, X, ls)
        #    b. mu = k_x @ alpha
        #    c. var = signal_var - k_x @ K_inv_kx (or similar)
        #    d. Compute EI
        # 5. Select best x, evaluate func, append to X, Y.
        
        # This is feasible if we vectorize the grid evaluation.
        
        # Let's refine the hyperparameters slightly.
        # We'll use a fixed length scale of 1.0 initially.
        # If the function is very oscillatory, we might need smaller ls.
        # Weierstrass f16: sum_{k=0}^infty a^k cos(b^k pi x)
        # a=0.5, b=3. The dominant frequency is from low k.
        # Period of cos(pi x) is 2. So length scale ~ 1.0 is reasonable.
        
        # Let's implement the BO loop.
        
        X_list = X.copy()
        Y_list = Y.copy()
        
        # Pre-compute grid points for acquisition function maximization
        # Use a grid of 4 points per dimension: 4^5 = 1024
        grid_vals = np.linspace(-5, 5, 4)
        grid = np.array(np.meshgrid(*[grid_vals]*dim)).T.reshape(-1, dim)
        
        # Add some random points to the grid to avoid missing narrow valleys?
        # Maybe mix grid and random.
        # Let's stick to grid for now.
        
        for i in range(budget - n_init):
            # Update GP model
            n = len(Y_list)
            K = compute_kernel(X_list, X_list, lengthscales)
            K += noise_var * np.eye(n)
            
            # Cholesky
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
            
            # Pre-compute alpha = K^{-1} Y
            # K alpha = Y => L L^T alpha = Y => L^T v = Y => L alpha = v
            v = np.linalg.solve(L, Y_list)
            alpha = np.linalg.solve(L.T, v)
            
            # Evaluate acquisition function on grid
            # k_grid: (n_grid, n)
            k_grid = compute_kernel(grid, X_list, lengthscales)
            
            # mu: (n_grid,)
            mu = k_grid @ alpha
            
            # var: (n_grid,)
            # var = signal_var - k_grid @ K^{-1} k_grid.T
            # K^{-1} = L^{-T} L^{-1}
            # Let's compute v_grid = L^{-1} k_grid.T  (n, n_grid)
            v_grid = np.linalg.solve(L, k_grid.T)
            var = signal_var - np.sum(v_grid * v_grid, axis=0)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            
            # Expected Improvement
            y_best = np.min(Y_list)
            xi = 0.01
            z = (mu - y_best - xi) / sigma
            ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Select best point
            idx = np.argmax(ei)
            x_new = grid[idx]
            
            # Evaluate
            y_new = func(x_new)
            
            # Append
            X_list = np.vstack([X_list, x_new])
            Y_list = np.append(Y_list, y_new)
            
            # Optional: adapt length scale?
            # For now, keep fixed.
            
        return X_list[-1], Y_list[-1]
# EVOLVE-BLOCK-END
