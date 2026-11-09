"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free surrogate-assisted search using a Gaussian Process."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Initial points: center + vertices of a hypercube
        center = np.zeros(self.dim)
        vertices = np.array([
            np.array([-5.0 if i % 2 == 0 else 5.0 for i in range(self.dim)] * 2)
            for _ in range(2)
        ])
        # Add some random points to cover more space
        rng = np.random.default_rng(42)
        rand_pts = rng.uniform(-5, 5, (10, self.dim))
        
        X = np.vstack([center, vertices, rand_pts])
        X = np.clip(X, -5, 5)
        
        Y = np.array([func(x) for x in X])
        
        best_x = X[np.argmin(Y)]
        best_y = np.min(Y)
        
        # Hyperparameters for GP
        # We use a simple RBF kernel: k(x, y) = sigma_f^2 * exp(-0.5 * ||x-y||^2 / l^2)
        # We will fix these for speed, or do a very quick optimization if time permits.
        # Given 500 budget and 5D, fixed reasonable hyperparameters are often sufficient.
        sigma_f = 10.0
        l = 2.0
        noise = 1e-6
        
        def kernel(X1, X2):
            # Squared distance
            sq_dists = np.sum(X1**2, 1)[:, None] + np.sum(X2**2, 1)[None, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / l**2)

        def gp_predict(X_new, X_train, Y_train):
            K = kernel(X_train, X_train) + noise * np.eye(len(X_train))
            K_s = kernel(X_train, X_new)
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            except np.linalg.LinAlgError:
                # Fallback if singular
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            K_new = kernel(X_new, X_train)
            mu = K_new @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        # Acquisition: Expected Improvement (EI)
        # We want to minimize, so EI = E[max(best_y - f(x), 0)]
        # For normal distribution N(mu, sigma):
        # EI = (best_y - mu) * Phi((best_y - mu)/sigma) + sigma * phi((best_y - mu)/sigma)
        # Note: standard EI is for maximization. For minimization, we flip signs or use -f.
        # Let's work with maximization of -f to use standard formulas, or derive for minimization.
        # Minimization EI:
        # If mu < best_y:
        #   z = (best_y - mu) / sigma
        #   EI = (best_y - mu) * Phi(z) + sigma * phi(z)
        # Else EI = 0
        
        from scipy.stats import norm

        def expected_improvement(mu, sigma, best_y):
            z = (best_y - mu) / sigma
            ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        # Main loop
        calls_made = len(X)
        
        for i in range(self.budget - calls_made):
            if time.time() > deadline:
                break
                
            # 1. Predict on a grid of candidate points
            # To keep it fast, we use a grid of candidate points rather than optimizing EI directly.
            # A grid of 5 points per dimension in 5D is 3125 points.
            # Kernel computation is O(N_train * N_cand * D).
            # N_train <= 500, N_cand = 3125, D = 5.
            # 500 * 3125 * 5 = ~7.8M ops per candidate set. Very fast.
            
            # Grid generation
            grid_vals = np.linspace(-5, 5, 11) # 11 points per dim -> 11^5 = 161051 points. Might be too many for late stages.
            # Let's use 7 points per dim -> 7^5 = 16807 points.
            grid_vals = np.linspace(-5, 5, 7)
            grid = np.array(np.meshgrid(*([grid_vals] * self.dim))).T.reshape(-1, self.dim)
            
            # Predict on grid
            mu, sigma = gp_predict(grid, X, Y)
            
            # Compute EI
            ei = expected_improvement(mu, sigma, best_y)
            
            # Select point with highest EI
            idx = np.argmax(ei)
            x_new = grid[idx]
            
            # Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new
            
            # Add to training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            
            # Optional: Update hyperparameters every 50 steps?
            # For now, fixed hyperparameters to ensure speed and stability.
            # The grid-based selection ensures we don't get stuck in local minima of the EI function itself.
            
        return best_x, best_y

# EVOLVE-BLOCK-END
