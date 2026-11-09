"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Gaussian Process Surrogate-Assisted Optimizer for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Data storage for GP
        X_list = []
        Y_list = []
        
        def add_point(x, f):
            nonlocal best_x, best_f, calls_used
            X_list.append(x)
            Y_list.append(f)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial Exploration
        # Use a small set of points to initialize the GP.
        # Random sampling is fine for 5D, but we can add a center point.
        n_init = 10
        if self.budget < n_init + 10:
            # Fallback if budget is tiny
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            add_point(x, fx)
            return best_x, best_f

        # Add center
        x_center = np.zeros(self.dim)
        fx_center = func(x_center)
        add_point(x_center, fx_center)
        
        for _ in range(n_init - 1):
            if time.time() > deadline or calls_used >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            add_point(x, fx)
        
        if not X_list:
            return best_x, best_f

        X = np.array(X_list)
        Y = np.array(Y_list)
        
        # GP Parameters
        # Rastrigin period is 2*pi ~ 6.28.
        # The function is smooth within the period.
        # Length scale should be smaller than the period to capture the wiggle, 
        # but large enough to be robust. 
        # Let's try L=1.0 or L=1.5. 
        # Noise: very small to avoid overfitting, but enough for stability.
        L = 1.2 
        sigma_f = 1.0 # Signal variance
        sigma_n = 1e-4 # Noise variance
        
        # Helper functions for GP
        def rbf_kernel(X1, X2, L):
            # X1: (n, d), X2: (m, d)
            # Periodic kernel? 
            # Standard RBF is not periodic. 
            # However, for a local search within a basin, RBF is fine.
            # If we want to exploit periodicity, we could use a periodic kernel:
            # k(x, y) = sigma_f^2 * exp( -2 * sum( sin^2(pi*(x_i-y_i)/period) ) / L^2 )
            # But the rotation/shift makes the "period" direction unknown.
            # So standard RBF is safer. It will just need to explore enough to find the basin.
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            return sigma_f**2 * np.exp(-sq_dist / (2 * L**2))

        def gp_predict(X_new):
            # Returns mean, var
            K = rbf_kernel(X, X, L) + sigma_n * np.eye(len(X))
            K_new = rbf_kernel(X, X_new, L)
            
            try:
                L_chol = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, Y))
                K_new_T = K_new.T
                v = np.linalg.solve(L_chol, K_new_T)
                mean = v.T @ alpha
                v2 = np.linalg.solve(L_chol, K_new_T)
                var = sigma_f**2 - np.sum(v2**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to simple inverse if Cholesky fails (rare with jitter)
                K += 1e-6 * np.eye(len(X))
                K_inv = np.linalg.inv(K)
                alpha = K_inv @ Y
                mean = K_new.T @ alpha
                var = sigma_f**2 - np.sum((K_new @ K_inv) * K_new.T, axis=0)
                var = np.maximum(var, 1e-10)
            
            return mean, var

        # Phase 2: Surrogate-Assisted Loop
        while calls_used < self.budget and time.time() < deadline:
            # Generate candidate points
            # We need a mix of exploration (high variance) and exploitation (low mean).
            # Expected Improvement (EI) acquisition:
            # EI(x) = (mu_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
            # where z = (mu_best - mu(x) - xi) / sigma(x)
            
            # To speed up, we can generate a batch of random candidates and pick the best EI.
            # Or use a grid? In 5D, grid is too expensive.
            # Random candidates: 50-100 per step is cheap.
            
            n_candidates = 100
            X_cand = rng.uniform(-5, 5, (n_candidates, self.dim))
            
            # Also include some points near the current best for exploitation
            if best_x is not None:
                # Add 20 candidates around best_x
                X_near = best_x + rng.normal(0, 0.5, (20, self.dim))
                X_near = np.clip(X_near, -5, 5)
                X_cand = np.vstack([X_cand, X_near])
            
            mu, var = gp_predict(X_cand)
            sigma = np.sqrt(var)
            
            # EI Calculation
            # xi is the exploration-exploitation balance parameter.
            # Start with a larger xi to explore, reduce later?
            # Or keep it constant.
            xi = 0.1 
            
            mu_best = best_f
            z = (mu_best - mu - xi) / sigma
            # Normal CDF and PDF
            # Using scipy if available, else approximate
            try:
                from scipy.stats import norm
                Phi = norm.cdf(z)
                phi = norm.pdf(z)
            except ImportError:
                # Approximate if scipy not available (should be available per prompt)
                # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
                # phi(z) = exp(-z^2/2) / sqrt(2*pi)
                Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
                phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            EI = (mu_best - mu - xi) * Phi + sigma * phi
            
            # Pick the candidate with max EI
            idx_best_cand = np.argmax(EI)
            x_next = X_cand[idx_best_cand]
            
            # Evaluate
            f_next = func(x_next)
            add_point(x_next, f_next)
            
            # Update X, Y arrays for next GP fit
            # To avoid O(N^3) growth issues with 500 points in 5D, 
            # 500x500 Cholesky is ~125M ops, which is fast in NumPy.
            # But we can also limit the GP memory to the last 100 points to keep it fast?
            # No, for 500 calls, full GP is fine.
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            # Optional: If we get stuck (no improvement for a while), 
            # we might want to restart the GP or increase exploration.
            # But EI naturally handles this by increasing variance in unexplored areas.

        return best_x, best_f
# EVOLVE-BLOCK-END
