"""Bayesian Optimization with GP surrogate for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

class Optimizer:
    """
    Optimizes Katsuura f23 using Bayesian Optimization with a Gaussian Process surrogate.
    1. Initialize with Latin Hypercube sampling to build an informative initial model.
    2. Use Expected Improvement acquisition function to balance exploration/exploitation.
    3. Periodically reseed with random points to escape local optima in the surrogate.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        
        calls_made = 0
        x_global_best = None
        f_global_best = np.inf
        
        def check_deadline():
            return time.time() > deadline
            
        def evaluate(x):
            nonlocal calls_made, x_global_best, f_global_best
            x_clipped = np.clip(x, lower, upper)
            fx = func(x_clipped)
            calls_made += 1
            if fx < f_global_best:
                f_global_best = fx
                x_global_best = x_clipped.copy()
            return x_clipped, fx

        # --- Latin Hypercube initialization ---
        n_init = min(25, max(10, budget // 20))
        
        def latin_hypercube(n, dim, lower, upper, rng):
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.random(n)) / n
                points[:, d] = lower + (upper - lower) * points[:, d]
            return points

        X_train = latin_hypercube(n_init, dim, lower, upper, rng)
        y_train = []
        
        for i in range(n_init):
            if calls_made >= budget or check_deadline():
                break
            x_pt = X_train[i]
            x_pt, f_pt = evaluate(x_pt)
            y_train.append(f_pt)
        
        if len(y_train) == 0:
            return x_global_best, f_global_best
            
        y_train = np.array(y_train)
        
        # GP parameters
        length_scale = 1.0
        noise = 1e-6
        signal_var = 1.0
        
        def rbf_kernel(X1, X2, length_scale, signal_var):
            """Compute RBF kernel matrix."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return signal_var * np.exp(-0.5 * sq_dists / length_scale**2)
        
        def gp_predict(X_test, X_train, y_train, length_scale, noise, signal_var):
            """Gaussian Process prediction: mean and variance."""
            K = rbf_kernel(X_train, X_train, length_scale, signal_var) + noise * np.eye(len(X_train))
            Ks = rbf_kernel(X_train, X_test, length_scale, signal_var)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mu = Ks.T @ alpha
                v = np.linalg.solve(L, Ks)
                var = signal_var - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: simple interpolation
                mu = np.full(X_test.shape[0], np.mean(y_train))
                var = np.full(X_test.shape[0], np.var(y_train) + 1e-10)
            
            return mu, var
        
        def expected_improvement(mu, var, f_best, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (f_best - mu - xi) / sigma
                ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(sigma > 1e-10, ei, 0.0)
            return ei
        
        # Main BO loop
        n_bo_steps = budget - calls_made
        bo_step = 0
        no_improve_count = 0
        f_best_prev = f_global_best
        
        while calls_made < budget and not check_deadline():
            bo_step += 1
            
            # Adaptive exploration parameter
            if bo_step < 30:
                xi = 0.05  # More exploration early
            elif bo_step < 100:
                xi = 0.02
            else:
                xi = 0.005  # More exploitation later
            
            # Generate candidate points for acquisition
            n_candidates = 500
            X_candidates = rng.uniform(lower, upper, size=(n_candidates, dim))
            
            # Add points near current best
            if x_global_best is not None:
                n_near = 100
                X_near = x_global_best + rng.normal(0, 0.5, size=(n_near, dim))
                X_near = np.clip(X_near, lower, upper)
                X_candidates = np.vstack([X_candidates, X_near])
            
            # Compute GP predictions
            mu, var = gp_predict(X_candidates, X_train, y_train, length_scale, noise, signal_var)
            
            # Compute EI
            ei = expected_improvement(mu, var, f_global_best, xi)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_next = X_candidates[best_idx]
            x_next, f_next = evaluate(x_next)
            
            # Update training set
            X_train = np.vstack([X_train, x_next.reshape(1, -1)])
            y_train = np.append(y_train, f_next)
            
            # Track improvement
            if f_next < f_best_prev - 1e-8:
                no_improve_count = 0
                f_best_prev = f_next
            else:
                no_improve_count += 1
            
            # Periodic reseeding to escape surrogate local optima
            if no_improve_count > 20 and calls_made < budget - 10:
                # Inject random points
                n_reseed = 5
                for _ in range(n_reseed):
                    if calls_made >= budget or check_deadline():
                        break
                    x_rand = rng.uniform(lower, upper, size=dim)
                    x_rand, f_rand = evaluate(x_rand)
                    X_train = np.vstack([X_train, x_rand.reshape(1, -1)])
                    y_train = np.append(y_train, f_rand)
                no_improve_count = 0
            
            # Update length scale adaptively
            if len(y_train) > 10:
                y_range = np.max(y_train) - np.min(y_train)
                if y_range > 1e-8:
                    # Reduce length scale as we get closer to optimum
                    progress = 1.0 - min(1.0, np.max(y_train) / max(1e-8, np.max(y_train[:n_init])))
                    length_scale = 1.0 * (1.0 - 0.5 * progress)
                    length_scale = max(0.3, length_scale)

        return x_global_best, f_global_best
# EVOLVE-BLOCK-END
