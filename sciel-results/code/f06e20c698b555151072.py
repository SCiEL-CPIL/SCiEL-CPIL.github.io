"""Bayesian Optimization with Gaussian Process surrogate for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        def gp_predict(X_train, y_train, X_test):
            """
            Gaussian Process prediction using RBF kernel.
            Returns mean and std for X_test.
            """
            n = X_train.shape[0]
            d = X_train.shape[1]
            
            # Normalize inputs to [0, 1] for better kernel behavior
            # Use min-max based on training data
            x_min = X_train.min(axis=0)
            x_max = X_train.max(axis=0)
            x_range = x_max - x_min
            x_range[x_range < 1e-10] = 1.0
            
            X_train_norm = (X_train - x_min) / x_range
            X_test_norm = (X_test - x_min) / x_range
            
            # RBF kernel with adaptive length scales
            # Use a single length scale for simplicity, or per-dimension
            # For n=5, let's use per-dimension length scales learned via simple heuristics
            # Actually, let's use a fixed reasonable length scale first
            # The box is [-5, 5], so range is 10. After normalization, range is 1.
            # Typical length scale for Weierstrass-like functions might be 0.1-0.5
            length_scale = 0.3
            signal_var = 1.0
            noise_var = 1e-8
            
            # Compute kernel matrix
            def rbf_kernel(A, B, ls):
                # A: (m, d), B: (p, d)
                sq_dists = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
                sq_dists = np.maximum(sq_dists, 0)
                return signal_var * np.exp(-0.5 * sq_dists / (ls ** 2))
            
            K = rbf_kernel(X_train_norm, X_train_norm, length_scale) + noise_var * np.eye(n)
            K_star = rbf_kernel(X_train_norm, X_test_norm, length_scale)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                v = np.linalg.solve(L, K_star.T)
            except np.linalg.LinAlgError:
                # Fallback to pinv
                alpha = np.linalg.pinv(K) @ y_train
                v = np.linalg.solve(K, K_star.T)
            
            mu = K_star @ alpha
            var = signal_var - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
            
            return mu, std

        def expected_improvement(mu, std, best_f, xi=0.01):
            """
            Compute Expected Improvement acquisition function.
            """
            if std < 1e-10:
                return np.zeros_like(mu)
            improvement = best_f - mu - xi
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return np.maximum(ei, 0)

        # Phase 1: Initial random sampling
        n_init = min(15, budget // 5)
        rng = np.random.default_rng(42)
        
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            call_func(x)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None

        # Collect training data
        # We need to store all evaluated points
        # Let's restructure: keep a list of (x, f)
        # But call_func doesn't return the history. Let's modify approach.
        # Actually, we can just re-evaluate? No, that wastes budget.
        # We need to store history. Let's add a list.
        
        # Wait, the current call_func doesn't store history.
        # I need to modify the structure to store history for GP.
        # Let's rewrite with explicit history storage.

        # RESTART WITH HISTORY STORAGE
        # Reset state
        best_x = None
        best_f = np.inf
        n_calls = 0
        history_x = []
        history_f = []

        def call_func_with_history(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                history_x.append(x_clipped.copy())
                history_f.append(f_val)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        # Phase 1: Initial random sampling
        n_init = min(15, budget // 5)
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            call_func_with_history(x)

        if best_x is None:
            x = np.zeros(dim)
            call_func_with_history(x)
            if best_x is None:
                return None

        # Phase 2: Bayesian Optimization loop
        # Use a subset of history for GP if it gets too large
        # For n=5 and budget=500, GP on up to 500 points is fine.
        
        while n_calls < budget and not check_time():
            # Select training data
            # Use all points if < 100, else sample or use all
            # For efficiency, use all points
            X_train = np.array(history_x)
            y_train = np.array(history_f)
            
            if len(X_train) < 2:
                # Need at least 2 points
                x = rng.uniform(-5, 5, dim)
                call_func_with_history(x)
                continue
            
            # Generate candidate points for acquisition
            # Use a grid or random sampling
            # Random sampling is faster and often sufficient
            n_candidates = 2000
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # GP prediction
            mu, std = gp_predict(X_train, y_train, X_cand)
            
            # Acquisition function
            ei = expected_improvement(mu, std, best_f)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            # Evaluate
            f_val, x_eval = call_func_with_history(x_next)
            if f_val is None:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
