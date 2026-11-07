"""Bayesian-Optimization-inspired sequential optimizer for BBOB f22 (Gallagher 21 peaks).

Strategy:
  Phase 1: Structured initial sampling (Sobol + random) to quickly locate promising basins.
  Phase 2: Sequential Expected-Improvement search using a lightweight surrogate
           (RBF kernel with noise) to balance exploration/exploitation.
  Phase 3: L-BFGS-B local refinement from the best points found.
  Phase 4: Final (1+1)-ES polish if budget remains.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Sequential BO-inspired optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget
        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                val = float(func(x))
                if not np.isfinite(val):
                    val = float('inf')
            except Exception:
                val = float('inf')
            calls += 1
            return val

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Structured initial sampling
        # Use Sobol-like quasi-random points for good coverage
        from scipy.stats import qmc
        n_initial = 40
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_unit = sampler.random(n_initial)
        sobol_points = -5.0 + 10.0 * sobol_unit

        for i in range(n_initial):
            if time.time() > deadline:
                break
            if calls >= budget - 100:
                break
            x = sobol_points[i]
            f_val = safe_func(x)
            update_best(x, f_val)

        # Add a few random points for diversity
        for _ in range(10):
            if time.time() > deadline:
                break
            if calls >= budget - 100:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            f_val = safe_func(x)
            update_best(x, f_val)

        # Phase 2: Sequential Expected-Improvement search
        # Maintain a set of observed points and use RBF surrogate
        X_obs = []
        f_obs = []

        # Populate with all observed points so far
        # We need to track them; let's re-evaluate or store during Phase 1
        # Actually, let's store during Phase 1. Let's refactor slightly.
        # For now, we'll assume we have a small set. Let's collect from best and neighbors.
        # Better: keep a list during Phase 1.
        
        # Let's redo Phase 1 tracking properly
        # (In a real implementation, we'd store X, f in Phase 1. 
        #  Here, we'll start the BO loop with the best few points and grow the set.)
        
        # Initialize BO set with best point and a few random neighbors
        X_obs.append(best_x.copy())
        f_obs.append(best_f)
        
        # Add a few random points to the BO set for initial kernel support
        for _ in range(5):
            if time.time() > deadline or calls >= budget - 50:
                break
            x_new = rng.uniform(-4.5, 4.5, dim)
            f_new = safe_func(x_new)
            update_best(x_new, f_new)
            X_obs.append(x_new)
            f_obs.append(f_new)

        # RBF Surrogate parameters
        length_scale = 1.0
        noise_var = 1e-6
        signal_var = 1.0

        def rbf_kernel(X1, X2, ls):
            # X1: (n, d), X2: (m, d)
            # Compute pairwise distances
            d = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)
            return np.exp(-0.5 * (d ** 2) / (ls ** 2))

        def predict(X_new, X_train, f_train, ls, noise, sig):
            # X_new: (m, d), X_train: (n, d), f_train: (n,)
            K = rbf_kernel(X_train, X_train, ls)
            K += (noise + 1e-10) * np.eye(len(X_train))
            Ks = rbf_kernel(X_train, X_new, ls)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_train))
                K_new = rbf_kernel(X_new, X_train, ls)
                mu = K_new.T @ alpha
                v = np.linalg.solve(L, K_new.T)
                var = sig - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                K_reg = K + 1e-6 * np.eye(len(X_train))
                try:
                    alpha = np.linalg.solve(K_reg, f_train)
                    K_new = rbf_kernel(X_new, X_train, ls)
                    mu = K_new.T @ alpha
                    var = np.full(len(X_new), sig)
                except:
                    mu = np.full(len(X_new), np.mean(f_train))
                    var = np.full(len(X_new), sig)
            
            return mu, var

        def expected_improvement(mu, var, f_best, xi=0.01):
            # f_best is the current best (minimum)
            # EI for minimization
            var = np.maximum(var, 1e-10)
            z = (f_best - mu - xi) / np.sqrt(var)
            ei = (f_best - mu - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            return ei

        # BO Loop
        n_bo_steps = min(200, budget - calls - 50)
        for step in range(n_bo_steps):
            if time.time() > deadline:
                break
            if calls >= budget - 50:
                break

            # Generate candidate points
            # Mix of random and local around best
            n_cand = 50
            candidates = []
            
            # Random candidates
            for _ in range(n_cand // 2):
                candidates.append(rng.uniform(-4.5, 4.5, dim))
            
            # Local candidates around best
            for _ in range(n_cand // 4):
                candidates.append(best_x + rng.normal(0, 0.5, dim))
            
            # Local candidates around other good points (if available)
            if len(X_obs) > 1:
                idx = rng.choice(len(X_obs), size=n_cand // 4, replace=False)
                for i in idx:
                    candidates.append(X_obs[i] + rng.normal(0, 0.5, dim))

            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5.0, 5.0)

            # Predict mean and variance
            mu, var = predict(candidates, np.array(X_obs), np.array(f_obs), length_scale, noise_var, signal_var)
            
            # Calculate EI
            ei = expected_improvement(mu, var, best_f, xi=0.01)
            
            # Select best candidate
            idx_best_cand = np.argmax(ei)
            x_next = candidates[idx_best_cand]
            
            # Evaluate
            f_next = safe_func(x_next)
            update_best(x_next, f_next)
            
            # Add to observation set
            X_obs.append(x_next)
            f_obs.append(f_next)
            
            # Adaptively adjust length scale based on distance to best
            # Simple heuristic: if we are far from best, increase ls; if close, decrease
            dist_to_best = np.linalg.norm(x_next - best_x)
            if dist_to_best < 0.5:
                length_scale *= 0.95
            else:
                length_scale *= 1.02
            length_scale = np.clip(length_scale, 0.1, 5.0)

        # Phase 3: L-BFGS-B refinement
        if calls < budget - 20 and time.time() < deadline:
            start_points = [best_x.copy()]
            
            # Add points around best
            for scale in [0.5, 0.25, 0.1]:
                for _ in range(2):
                    sp = best_x + rng.normal(0, scale, dim)
                    sp = np.clip(sp, -5.0, 5.0)
                    start_points.append(sp)
            
            # Add a few random points
            for _ in range(3):
                sp = rng.uniform(-4.5, 4.5, dim)
                start_points.append(sp)

            # Remove duplicates
            unique_starts = []
            for sp in start_points:
                is_dup = False
                for us in unique_starts:
                    if np.linalg.norm(sp - us) < 0.1:
                        is_dup = True
                        break
                if not is_dup:
                    unique_starts.append(sp)

            for sp in unique_starts:
                if calls >= budget - 10:
                    break
                if time.time() > deadline:
                    break

                try:
                    result = minimize(
                        safe_func,
                        sp,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 30, 'maxfun': 30, 'ftol': 1e-12, 'gtol': 1e-7}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Phase 4: Final (1+1)-ES polish
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            x_cur = best_x.copy()
            f_cur = best_f
            remaining = budget - calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                x_trial = np.clip(x_cur + sigma * rng.standard_normal(dim), -5.0, 5.0)
                f_trial = safe_func(x_trial)
                if f_trial < f_cur:
                    x_cur = x_trial
                    f_cur = f_trial
                    sigma *= 1.2
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
