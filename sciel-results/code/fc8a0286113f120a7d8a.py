import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        def _time_ok():
            return time.time() - start_time < time_limit

        rng = np.random.default_rng(42)

        # Data storage
        X = []
        Y = []

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not _time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if len(x) != dim:
                x = x[:dim]
            try:
                f = func(x)
            except Exception:
                return None
            calls += 1
            X.append(x.copy())
            Y.append(float(f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Sobol-like sampling for diversity
        n_init = min(20, budget)
        for i in range(n_init):
            if not _time_ok() or calls >= budget:
                break
            # Use a quasi-random sequence for better coverage than pure random
            # Simple Halton-like or just stratified random
            # For simplicity and speed, use random but ensure spread
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)

        if best_x is None:
            # Fallback if something went wrong
            x = np.zeros(dim)
            safe_eval(x)
            return best_x, best_f

        # Phase 2: Bayesian Optimization with GP
        # We will use a simple GP implementation with RBF kernel
        # Since SciPy is available, we could use a library, but let's implement a lightweight one
        # or use a simpler surrogate if GP is too slow.
        # Given dim=5 and budget=300, a full GP might be slow if not optimized.
        # Let's try a lightweight GP.

        def gp_predict(X_train, Y_train, X_test, length_scale=1.0, noise=1e-8):
            # X_train: (n, d), Y_train: (n,), X_test: (m, d)
            # Returns mean, std at X_test
            
            # Normalize Y for numerical stability
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std == 0:
                y_std = 1.0
            Y_norm = (Y_train - y_mean) / y_std

            # RBF Kernel
            def rbf_kernel(A, B, l):
                # A: (n, d), B: (m, d)
                # dist^2 = sum(A^2) + sum(B^2) - 2AB^T
                A2 = np.sum(A**2, axis=1)[:, np.newaxis]
                B2 = np.sum(B**2, axis=1)[:, np.newaxis]
                D2 = A2 + B2.T - 2 * np.dot(A, B.T)
                D2 = np.maximum(D2, 0)
                return np.exp(-D2 / (2 * l**2))

            n = len(X_train)
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
            K_test = rbf_kernel(X_train, X_test, length_scale)
            
            # Solve K alpha = Y_norm
            try:
                alpha = np.linalg.solve(K, Y_norm)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
            
            mean = y_mean + y_std * np.dot(K_test.T, alpha)
            
            # Variance
            v = rbf_kernel(X_test, X_test, length_scale)
            var = np.diag(v) - np.sum(K_test.T * alpha[:, np.newaxis], axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            return mean, std

        # Phase 2: BO Loop
        # We will run BO until budget is exhausted or time runs out
        # To handle the "early progress counts" metric, we should keep the best found so far.
        
        # Adaptive length scale? Start with 1.0, maybe adjust based on best improvement.
        # For simplicity, keep fixed length scale or a few presets.
        # Let's use a fixed length scale of 1.0 initially.
        
        # To speed up GP, we can limit the number of training points used if it gets too large,
        # but 300 points in 5D is small, so full GP should be fine.
        
        # Candidate generation for EI maximization
        # We will use a combination of:
        # 1. Random candidates
        # 2. Perturbations of current best
        # 3. Perturbations of top-k points
        
        top_k = 5
        
        while calls < budget and _time_ok():
            # Get top k points
            if len(Y) > top_k:
                top_indices = np.argsort(Y)[:top_k]
                top_points = np.array([X[i] for i in top_indices])
                top_values = np.array([Y[i] for i in top_indices])
            else:
                top_points = np.array(X)
                top_values = np.array(Y)
            
            best_val = np.min(Y)
            
            # Generate candidates
            candidates = []
            
            # 1. Random candidates
            n_random = 50
            for _ in range(n_random):
                candidates.append(rng.uniform(-5, 5, dim))
            
            # 2. Local perturbations of top points
            n_local = 50
            for i in range(n_local):
                # Pick a random top point
                idx = rng.integers(0, len(top_points))
                x_base = top_points[idx]
                # Step size decreases as we get closer to optimum?
                # Use a fixed small step for exploration around good points
                step = 0.5
                x_new = x_base + step * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
                candidates.append(x_new)
            
            # 3. Interpolation between top points
            n_interp = 20
            for _ in range(n_interp):
                if len(top_points) >= 2:
                    i1 = rng.integers(0, len(top_points))
                    i2 = rng.integers(0, len(top_points))
                    if i1 != i2:
                        t = rng.random()
                        x_new = (1 - t) * top_points[i1] + t * top_points[i2]
                        x_new = np.clip(x_new, -5, 5)
                        candidates.append(x_new)
            
            candidates = np.array(candidates)
            
            # Evaluate GP on candidates
            # This might be slow if len(X) is large, but n=300 is okay.
            # However, doing this every iteration might be too slow.
            # Let's check time.
            
            if len(X) > 5: # Need at least a few points for GP
                try:
                    mean, std = gp_predict(np.array(X), np.array(Y), candidates, length_scale=1.0, noise=1e-8)
                    
                    # Expected Improvement
                    # EI = (mu - best_val + xi) * Phi(z) + sigma * phi(z)
                    # where z = (mu - best_val + xi) / sigma
                    xi = 0.01 # Small exploration bonus
                    improvement = best_val - mean # We want to minimize, so improvement is best - mean
                    # Note: Standard EI is for maximization. For minimization:
                    # EI = (best_val - mu) * Phi(z) + sigma * phi(z)
                    # where z = (best_val - mu) / sigma
                    # If best_val < mu, improvement is negative, EI should be 0 or small.
                    
                    z = (best_val - mean) / (std + 1e-10)
                    EI = (best_val - mean) * norm.cdf(z) + std * norm.pdf(z)
                    
                    # Select best candidate
                    idx_best = np.argmax(EI)
                    x_next = candidates[idx_best]
                except Exception:
                    # Fallback to random if GP fails
                    x_next = candidates[0]
            else:
                # Not enough points for GP, just pick random
                x_next = candidates[0]
            
            # Evaluate
            f_new = safe_eval(x_next)
            if f_new is None:
                break
        
        # Phase 3: Final Polish
        # Use remaining budget for a very fine local search around the best point
        if best_x is not None and calls < budget and _time_ok():
            remaining = budget - calls
            # Do a simple hill climbing with very small steps
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if not _time_ok() or calls >= budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = max(sigma, 0.001)
                sigma = min(sigma, 0.5)
        
        return best_x, best_f
