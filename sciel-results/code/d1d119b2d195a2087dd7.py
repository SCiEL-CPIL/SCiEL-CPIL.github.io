"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Bayesian Optimizer with Matern-5/2 GP for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        
        # Domain bounds
        low = -5.0 * np.ones(dim)
        high = 5.0 * np.ones(dim)
        
        # Initial points: 20 points
        # Mix of center and random
        X = []
        y = []
        
        # Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        X.append(x0)
        y.append(f0)
        calls_made = 1
        
        # Random points
        for _ in range(19):
            if calls_made >= budget:
                break
            x = rng.uniform(low, high)
            f = func(x)
            X.append(x)
            y.append(f)
            calls_made += 1
            
        X = np.array(X)
        y = np.array(y)
        
        best_f = np.min(y)
        best_x = X[np.argmin(y)].copy()
        
        # GP Parameters
        # Matern 5/2 kernel
        # K(x, x') = sigma_f^2 * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
        # where d = ||x - x'|| / l
        # We use a fixed length scale initially, but could adapt.
        # Given the oscillatory nature of Weierstrass, length scale matters.
        # Weierstrass f16 has frequencies increasing. 
        # Let's try a moderate length scale, e.g., 1.0 or 0.5.
        # Actually, since the function is highly oscillatory, a shorter length scale might be better 
        # to capture local variations, but GP might struggle with high frequency if l is too small 
        # relative to the distance between points.
        # Let's start with l=1.0 and sigma_f=1.0.
        
        length_scale = 1.0
        sigma_f = 1.0
        noise = 1e-6
        
        # Acquisition: Expected Improvement (EI)
        # EI(x) = (mu(x) - f_best - xi) * Phi((mu(x) - f_best - xi) / sigma(x)) + sigma(x) * phi((mu(x) - f_best - xi) / sigma(x))
        
        # For efficiency, we will use a grid-based search for the acquisition function maximization.
        # A full grid is too expensive. We'll use a sparse grid or random sampling + local optimization.
        # Given dim=5, a grid of 5^5 = 3125 points is feasible for evaluation of GP and EI.
        # Let's use a grid of size 5 per dimension -> 3125 points.
        # Or 7^5 = 16807. 16k GP evaluations is fast (matrix ops).
        # Let's use 7 points per dimension.
        
        grid_size = 7
        # Create grid
        grid_vals = np.linspace(low, high, grid_size)
        grid_points = np.meshgrid(*[grid_vals]*dim)
        grid_points = np.vstack([gp.ravel() for gp in grid_points]).T
        
        # Pre-compute distances from grid points to initial points? No, X changes.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # 1. Fit GP to current X, y
            # We need to compute K(X, X) + noise*I
            # And K(X, X_new) for the grid
            
            # Kernel function
            def kernel(X1, X2, l):
                # X1: (n1, d), X2: (n2, d)
                # Compute squared distances
                X1_sq = np.sum(X1**2, axis=1)[:, None]
                X2_sq = np.sum(X2**2, alpha=1)[None, :]
                # Correct:
                X2_sq = np.sum(X2**2, axis=1)[None, :]
                
                D_sq = np.maximum(X1_sq + X2_sq - 2 * X1 @ X2.T, 0)
                D = np.sqrt(D_sq)
                
                # Matern 5/2
                sqrt_5_D = np.sqrt(5) * D / l
                K = (sigma_f**2) * (1 + sqrt_5_D + (5/3) * D_sq / (l**2)) * np.exp(-sqrt_5_D)
                return K
            
            # Fit
            K = kernel(X, X, length_scale)
            K += noise * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            except np.linalg.LinAlgError:
                # Fallback if singular
                K += 1e-4 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                except:
                    break # GP failed, break
            
            # 2. Compute GP posterior on grid
            K_grid = kernel(X, grid_points, length_scale)
            
            # Mean
            mu_grid = K_grid @ alpha
            
            # Variance
            # Var(x) = K(x,x) - K(x, X) K(X,X)^{-1} K(X, x)
            # K(x,x) is sigma_f^2 for all x (if no noise in prior, but we added noise to training)
            # Actually, K(x,x) = sigma_f^2.
            v = np.linalg.solve(L, K_grid.T)
            var_grid = sigma_f**2 - np.sum(v**2, axis=0)
            var_grid = np.maximum(var_grid, 1e-8)
            std_grid = np.sqrt(var_grid)
            
            # 3. Compute EI
            # f_best is the minimum so far
            # EI = (f_best - mu - xi) * Phi((f_best - mu - xi) / std) + std * phi((f_best - mu - xi) / std)
            # Note: We are minimizing.
            # Standard EI for minimization:
            # Let d = (f_best - mu - xi) / std
            # EI = (f_best - mu - xi) * Phi(d) + std * phi(d)
            
            xi = 0.01 * (np.max(y) - np.min(y) + 1e-6) # Exploration parameter
            
            d = (best_f - mu_grid - xi) / std_grid
            
            # Normal CDF and PDF
            # Use scipy if available? Task says SciPy is available.
            from scipy.stats import norm
            Phi = norm.cdf(d)
            phi = norm.pdf(d)
            
            EI = (best_f - mu_grid - xi) * Phi + std_grid * phi
            
            # 4. Select next point
            idx_best = np.argmax(EI)
            x_next = grid_points[idx_best].copy()
            
            # Optional: Local refinement around x_next?
            # With 500 budget, we can afford some local search.
            # But let's stick to the grid for simplicity and robustness first.
            # If we have time, we can do a few random points near x_next.
            
            # 5. Evaluate
            f_next = func(x_next)
            calls_made += 1
            
            # Update
            X = np.vstack([X, x_next])
            y = np.append(y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
                
            # Adaptive Length Scale?
            # If we are stuck, maybe change length scale?
            # For now, keep it fixed.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
