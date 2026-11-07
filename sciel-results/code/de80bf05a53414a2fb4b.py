"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """
    Bayesian Optimization Optimizer for BBOB functions.
    Uses a Gaussian Process surrogate with Expected Improvement acquisition.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        # State
        X = []  # List of points
        Y = []  # List of function values
        x_best = None
        f_best = float('inf')
        calls = 0

        # Surrogate parameters
        # We will use a simple GP implementation with RBF kernel.
        # For dim=5, n=300, O(n^3) is 27 million ops per fit, which is fast in NumPy.
        
        def safe_call(x):
            nonlocal calls, x_best, f_best, X, Y
            if time.time() > deadline or calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            if val < f_best:
                f_best = val
                x_best = x_clipped.copy()
            X.append(x_clipped)
            Y.append(val)
            return val

        # Phase 1: Initial Sampling
        # Use a mix of center, corners, and random points to ensure coverage.
        # For 5D, 3^5 = 243 is too many. Let's use 10-20 points.
        n_init = 15
        rng = np.random.default_rng(42)
        
        # Center
        init_points = [np.zeros(dim)]
        # Random
        random_points = rng.uniform(-5, 5, (n_init - 1, dim))
        for p in random_points:
            init_points.append(p)
            
        for p in init_points:
            if time.time() > deadline or calls >= budget:
                break
            val = safe_call(p)
            if val is None:
                break

        if x_best is None:
             # Fallback if no calls were made (should not happen)
             x_best = np.zeros(dim)
             f_best = 0.0
             return x_best, f_best

        # Phase 2: Bayesian Optimization Loop
        # We will iterate until budget or time is exhausted.
        
        # GP Kernel: RBF (Squared Exponential)
        # k(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        # We fix length scale l=1.0 and signal variance sigma_f^2=1.0 for simplicity,
        # or we can try to optimize them, but that adds complexity and time.
        # Given the budget, fixed hyperparameters are often robust enough for BO.
        
        def compute_kernel(X1, X2, l=1.0, sigma_f=1.0):
            # X1: (n, d), X2: (m, d)
            # Returns (n, m) kernel matrix
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (l**2))

        def fit_gp(X_train, Y_train):
            """
            Fits a simple GP. Returns mean and std functions.
            Uses fixed hyperparameters: l=1.0, sigma_f=1.0, noise=1e-6.
            """
            n = len(X_train)
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            
            # Normalize Y for numerical stability? 
            # BBOB functions can have large offsets. Let's center Y.
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y_train - y_mean) / y_std
            
            K = compute_kernel(X_train, X_train)
            noise = 1e-6
            K += noise * np.eye(n)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                try:
                    alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
                    # For prediction, we need K_inv. We can just use lstsq for K_inv @ x
                    # But let's stick to Cholesky for speed if possible.
                    # If it fails, we might have degenerate points.
                    # Let's add jitter.
                    K += 1e-4 * np.eye(n)
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                except Exception:
                    # Return constant prediction
                    return (lambda x: y_mean), (lambda x: 0.0), y_mean, y_std

            def predict(X_new):
                """
                Returns mean and std for X_new.
                """
                X_new = np.array(X_new)
                if X_new.ndim == 1:
                    X_new = X_new.reshape(1, -1)
                
                K_star = compute_kernel(X_new, X_train)
                mean_norm = K_star @ alpha
                mean = mean_norm * y_std + y_mean
                
                # Variance
                # Var = sigma_f^2 - K_star @ K_inv @ K_star.T
                # We can compute K_inv @ K_star.T using L
                v = np.linalg.solve(L, K_star.T)
                var_norm = 1.0 - np.sum(v * K_star, axis=1)
                var = np.maximum(var_norm, 0) * y_std**2
                std = np.sqrt(var)
                
                return mean, std

            return predict, y_mean, y_std

        # Acquisition Function: Expected Improvement (EI)
        def expected_improvement(x, f_best_norm, predict_func, y_mean, y_std):
            """
            Calculates EI for a batch of points x.
            x: (n, d)
            Returns array of EI values.
            """
            mean, std = predict_func(x)
            # Normalize f_best to match GP scale
            f_best_norm_val = (f_best - y_mean) / y_std
            
            # Z = (mean - f_best_norm) / std
            # Handle std=0
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (mean - f_best_norm_val) / std
                Z = np.where(std < 1e-8, 0, Z)
            
            # EI = Z * phi(Z) + phi(Z)
            # phi is standard normal pdf
            # For Z < 0, EI = 0
            # For Z >= 0, EI = Z * phi(Z) + phi(Z)
            from scipy.stats import norm
            EI = np.zeros_like(mean)
            mask = Z > 0
            EI[mask] = Z[mask] * norm.pdf(Z[mask]) + norm.cdf(Z[mask])
            # Scale back to original space? 
            # EI is usually computed in normalized space. 
            # To compare points, we can just use the normalized EI.
            return EI

        # Candidate Generation Strategy
        # Instead of sampling random points and checking EI, we generate a set of candidates
        # and pick the best.
        # Candidates:
        # 1. Local perturbations around x_best (Gaussian)
        # 2. Random points in the box
        # 3. Points on a coarse grid (optional, might be too many)
        
        # We will generate ~100-200 candidates per iteration.
        
        while calls < budget and time.time() < deadline:
            
            # Fit GP on current data
            if len(X) < 2:
                # Need at least 2 points for GP to be meaningful
                x_next = rng.uniform(-5, 5, dim)
                val = safe_call(x_next)
                if val is None: break
                continue
                
            try:
                predict_func, y_mean, y_std = fit_gp(X, Y)
            except Exception:
                # If GP fails, fallback to random
                x_next = rng.uniform(-5, 5, dim)
                val = safe_call(x_next)
                if val is None: break
                continue

            # Generate Candidates
            n_candidates = 100
            candidates = []
            
            # 1. Local exploration around best
            # Sigma decreases as we find better solutions? 
            # Simple adaptive sigma: start at 1.0, shrink if no improvement.
            # For simplicity, use a fixed mix.
            sigma_local = 0.5
            local_candidates = x_best + rng.normal(0, sigma_local, (n_candidates // 2, dim))
            local_candidates = np.clip(local_candidates, -5, 5)
            candidates.append(local_candidates)
            
            # 2. Global exploration
            global_candidates = rng.uniform(-5, 5, (n_candidates // 2, dim))
            candidates.append(global_candidates)
            
            # 3. Maybe some points near other good points?
            # Let's stick to best + random for now.
            
            candidates = np.vstack(candidates)
            
            # Evaluate EI for all candidates
            ei_values = expected_improvement(candidates, f_best, predict_func, y_mean, y_std)
            
            # Pick the candidate with max EI
            idx_best = np.argmax(ei_values)
            x_next = candidates[idx_best]
            
            # Evaluate
            val = safe_call(x_next)
            if val is None:
                break

            # Optional: If we have budget left, we could do a local polish with Nelder-Mead
            # but BO is already doing local exploitation via EI.
            # However, BO is slow per step (O(n^3)).
            # With 300 budget, we do ~285 BO steps.
            # 285 * (300^3) ~ 7.7e9 ops. This might be too slow in Python/NumPy.
            # Let's check timing.
            # Actually, 300^3 is 27,000,000.
            # 285 steps * 27M = 7.7 billion flops.
            # NumPy is fast, but 7.7B flops might take 5-10 seconds.
            # Total time limit 300s. This should be fine.
            
            # Safety check: if we are running out of time, stop BO and do a final local polish?
            # Or just stop.
            if time.time() > deadline - 5:
                break

        return x_best, f_best

# EVOLVE-BLOCK-END
