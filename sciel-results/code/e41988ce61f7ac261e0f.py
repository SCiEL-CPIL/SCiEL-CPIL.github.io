"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
import math

class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Data storage for GP
        X = []
        Y = []

        def tracked_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Helper: Compute GP mean and variance
        def gp_predict(X_train, Y_train, X_pred):
            # Standardize Y for numerical stability
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1e-8
            Y_std = (Y_train - y_mean) / y_std
            
            # Kernel: Matern 5/2
            def kernel(X1, X2):
                # X1: (n, d), X2: (m, d)
                # Compute squared distances
                X1_sq = np.sum(X1**2, axis=1)
                X2_sq = np.sum(X2**2, axis=1)
                dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
                dist_sq = np.maximum(dist_sq, 0)
                dist = np.sqrt(dist_sq)
                # Matern 5/2: (1 + sqrt(3)*d + 3/5*d^2) * exp(-sqrt(3)*d)
                sqrt3_d = np.sqrt(3) * dist
                k = (1 + sqrt3_d + 0.6 * dist_sq) * np.exp(-sqrt3_d)
                return k

            K = kernel(X_train, X_train)
            # Add small jitter for numerical stability
            K += 1e-6 * np.eye(len(X_train))
            
            # Precompute inverse
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # Fallback if singular (should be rare with jitter)
                K_inv = np.linalg.pinv(K)
                
            K_star = kernel(X_train, X_pred)
            
            # Mean and Variance
            mu = K_star.T @ K_inv @ Y_std
            v = kernel(X_pred, X_pred)
            var = v - K_star.T @ K_inv @ K_star
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            # Unstandardize
            mu = mu * y_std + y_mean
            std = std * y_std
            
            return mu, std

        # Helper: Calculate Expected Improvement
        def expected_improvement(mu, std, f_best, xi=0.01):
            # EI = (f_best - f) * Phi((f_best - f) / sigma) + sigma * phi((f_best - f) / sigma)
            # where f is the predicted mean (mu)
            # Note: We are minimizing.
            # Standard EI for minimization:
            # Let Z = (f_best - mu) / sigma
            # EI = sigma * (Z * Phi(Z) + phi(Z))
            # If sigma is 0, EI is 0
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (f_best - mu) / std
                ei = std * (Z * norm.cdf(Z) + norm.pdf(Z))
            ei = np.where(std > 1e-8, ei, 0.0)
            return ei

        # Phase 1: Initial Latin Hypercube Sampling (LHS)
        # 20 points for initial exploration
        n_init = min(20, self.budget // 10)
        
        # Generate LHS points in [0, 1]^d
        n = n_init
        d = self.dim
        # Simple LHS: divide each dimension into n intervals, shuffle indices
        lhs = np.zeros((n, d))
        for j in range(d):
            intervals = (np.arange(n) + rng.random(n)) / n
            rng.shuffle(intervals)
            lhs[:, j] = intervals
        
        # Map to [-5, 5]^d
        X_init = lhs * 10 - 5
        
        for x in X_init:
            if time.time() > deadline or calls_made >= self.budget:
                break
            fx = tracked_func(x)
            X.append(x)
            Y.append(fx)
            
        if best_x is None:
            return None, np.inf

        # Phase 2: Bayesian Optimization Loop
        # Use a grid of candidates for EI maximization to avoid expensive local optimization of EI
        # Grid resolution: 20 points per dimension -> 20^5 = 3.2M points. This is too large.
        # Better: Use a smaller grid or random candidates.
        # Let's use a grid of 10 points per dimension -> 10^5 = 100,000 candidates.
        # This is manageable for GP prediction on 5D.
        
        grid_size = 10
        grid_vals = np.linspace(-5, 5, grid_size)
        # Generate full grid
        # This creates an array of shape (100000, 5)
        # Using np.meshgrid
        grids = np.meshgrid(*([grid_vals] * d), indexing='ij')
        X_grid = np.array(grids).T.reshape(-1, d)
        
        # We will update the candidate set occasionally or just use the static grid.
        # Static grid is simpler and deterministic.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Ensure we have enough points to build a meaningful GP
            # At least d+1 points, but let's wait for at least 10 or so
            if len(X) < 10:
                # If we haven't filled the init phase (shouldn't happen with n_init=20),
                # or if we are in a weird state, just sample random
                x = rng.uniform(-5, 5, d)
                fx = tracked_func(x)
                X.append(x)
                Y.append(fx)
                continue

            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Predict on grid
            try:
                mu, std = gp_predict(X_arr, Y_arr, X_grid)
            except Exception:
                # If GP fails, fallback to random
                x = rng.uniform(-5, 5, d)
                fx = tracked_func(x)
                X.append(x)
                Y.append(fx)
                continue

            # Calculate EI
            ei = expected_improvement(mu, std, best_f)
            
            # Select point with max EI
            idx_max = np.argmax(ei)
            x_next = X_grid[idx_max]
            
            # Evaluate
            fx = tracked_func(x_next)
            X.append(x_next)
            Y.append(fx)
            
            # Optional: If we get stuck (no improvement in last K iterations),
            # we might want to explore more. But EI handles this via the exploration term.
            # However, if the GP is overconfident, we might loop.
            # A simple check: if EI is very low everywhere, pick random.
            if np.max(ei) < 1e-4:
                # Explore randomly
                x_next = rng.uniform(-5, 5, d)
                fx = tracked_func(x_next)
                X.append(x_next)
                Y.append(fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
