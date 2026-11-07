import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization with Gaussian Process for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Initial random sampling to build GP prior
        n_init = 30
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(np.clip(x, -5, 5)) for x in X])
        calls_made += n_init

        best_idx = np.argmin(Y)
        best_f = Y[best_idx]
        best_x = X[best_idx].copy()

        # Phase 2: Bayesian Optimization with GP and Expected Improvement
        # GP Hyperparameters (tuned for smooth-ish landscape)
        # We use a Matern 5/2 kernel for better behavior on non-analytic functions
        length_scale = 1.5
        variance = 10.0
        noise = 1e-6

        def matern52_kernel(X1, X2, length_scale, variance):
            dist = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
            dist = np.maximum(dist, 1e-10)
            sqrt3_dist = np.sqrt(3) * dist / length_scale
            K = variance * (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
            return K

        def compute_gp_posterior(X_obs, Y_obs, X_new, length_scale, variance, noise):
            K = matern52_kernel(X_obs, X_obs, length_scale, variance) + noise * np.eye(len(X_obs))
            Ks = matern52_kernel(X_obs, X_new, length_scale, variance)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(len(K))
                L = np.linalg.cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            mu = Ks.T @ alpha
            v = np.linalg.solve(L, Ks)
            cov = variance - v.T @ v
            var = np.diag(cov)
            var = np.maximum(var, 1e-10)
            return mu, var

        # Acquisition Function: Expected Improvement (EI)
        # We use a local search around the best point to find the next candidate
        # to avoid the high cost of global optimization in high dimensions.
        
        n_bo = budget - n_init
        batch_size = 1  # Sequential BO

        for step in range(n_bo):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Adaptive length scale: shrink as we converge
            # If best_f is close to 0, we are likely in a local minimum, shrink scale
            # If best_f is large, we are far, keep scale larger
            current_scale = length_scale
            if best_f < 1.0:
                current_scale = 0.5
            elif best_f < 10.0:
                current_scale = 1.0
            else:
                current_scale = 2.0

            # Find the next point using local optimization around current best
            # We sample candidate points in a neighborhood of the best point
            # and also some random points to allow escape from local minima
            
            candidates = []
            # 50% local, 50% global exploration
            n_local = 25
            n_global = 25
            
            # Local candidates around best_x
            local_radius = current_scale * 1.5
            for _ in range(n_local):
                offset = rng.normal(0, local_radius, size=dim)
                x_cand = best_x + offset
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # Global candidates
            for _ in range(n_global):
                x_cand = rng.uniform(-5, 5, size=dim)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Compute GP posterior for candidates
            mu, var = compute_gp_posterior(X, Y, candidates, current_scale, variance, noise)
            std = np.sqrt(var)
            
            # Expected Improvement
            # EI = (f_best - mu - xi) * Phi((f_best - mu - xi)/std) + std * phi((f_best - mu - xi)/std)
            # We use xi=0 for simplicity, or a small xi for exploration
            xi = 0.01
            z = (best_f - mu - xi) / std
            ei = (best_f - mu - xi) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select candidate with max EI
            best_cand_idx = np.argmax(ei)
            x_next = candidates[best_cand_idx]
            
            # Evaluate
            f_next = func(np.clip(x_next, -5, 5))
            calls_made += 1
            
            # Update data
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
                
            # Periodically reset GP if it gets too large (memory/compute)
            # Keep only the best 50 points? No, GP needs all points for accuracy.
            # But 500 points in 5D is manageable.
            
            # Optional: Hyperparameter adaptation
            # If we are stuck (no improvement for many steps), increase exploration
            # We can track stagnation
            if step % 50 == 0 and step > 0:
                # Re-initialize length scale if stuck
                pass

        return best_f
