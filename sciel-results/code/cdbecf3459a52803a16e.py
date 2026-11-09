import numpy as np
import time
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    """Multi-Start L-BFGS-B + Valley-Following Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0
        self._cache = {}

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0
        self._cache = {}

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            key = tuple(x.tolist())
            if key in self._cache:
                return self._cache[key]
            
            self.calls_made += 1
            val = func(x)
            self._cache[key] = val
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def valley_follow(x0, max_iters=20):
            """Budget-aware adaptive valley-following for Rosenbrock."""
            x = np.clip(x0.copy(), -5, 5)
            f = wrapped_func(x)
            step_size = 0.5
            
            for it in range(max_iters):
                if self.calls_made >= self.budget - 5:
                    break
                if time.time() > self.deadline:
                    break
                
                improved = False
                
                # Adaptive valley-following: blend toward x[i+1] = x[i]^2
                for i in range(self.dim - 1):
                    if self.calls_made >= self.budget - 5:
                        break
                    target_next = x[i] ** 2
                    blend = 0.7  # strong pull toward valley
                    x_vf = x.copy()
                    x_vf[i+1] = (1 - blend) * x[i+1] + blend * target_next
                    x_vf = np.clip(x_vf, -5, 5)
                    f_vf = wrapped_func(x_vf)
                    if f_vf < f:
                        x = x_vf
                        f = f_vf
                        improved = True
                        step_size = min(step_size * 1.2, 1.0)
                    else:
                        step_size = max(step_size * 0.7, 0.05)
                
                # Adaptive coordinate descent with golden-section-like steps
                for i in range(self.dim):
                    if self.calls_made >= self.budget - 5:
                        break
                    best_local_f = f
                    best_local_x = x.copy()
                    for step_mult in [step_size, step_size * 0.5, step_size * 0.25]:
                        if self.calls_made >= self.budget - 5:
                            break
                        for direction in [1.0, -1.0]:
                            if self.calls_made >= self.budget - 5:
                                break
                            x_new = x.copy()
                            x_new[i] += step_mult * direction
                            x_new = np.clip(x_new, -5, 5)
                            f_new = wrapped_func(x_new)
                            if f_new < best_local_f:
                                best_local_f = f_new
                                best_local_x = x_new
                    if best_local_f < f:
                        x = best_local_x
                        f = best_local_f
                        improved = True
                
                if not improved:
                    step_size = max(step_size * 0.5, 0.01)
                    if step_size < 0.02:
                        break
            
            return x, f

        def gp_surrogate_minimize(X_train, y_train, n_candidates=50):
            """
            Fit a simple GP surrogate and find the most promising point
            by minimizing the posterior mean (exploitation-focused).
            Returns the best candidate point.
            """
            n = len(X_train)
            d = self.dim
            
            # Normalize inputs to [0, 1] for better GP conditioning
            X_min = X_train.min(axis=0)
            X_max = X_train.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range < 1e-10] = 1.0
            X_norm = (X_train - X_min) / X_range
            
            # Normalize outputs
            y_mean = y_train.mean()
            y_std = y_train.std()
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y_train - y_mean) / y_std
            
            # Simple RBF kernel with fixed hyperparameters
            # Length scale: try a few values and pick best via simple criterion
            best_length_scale = 0.3
            best_score = np.inf
            
            for ls in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                # Compute kernel matrix
                sq_dists = np.sum(X_norm ** 2, axis=1)[:, None] + \
                           np.sum(X_norm ** 2, axis=1)[None, :] - \
                           2 * X_norm @ X_norm.T
                sq_dists = np.maximum(sq_dists, 0)
                K = np.exp(-sq_dists / (2 * ls ** 2))
                
                # Add small jitter for numerical stability
                K += 1e-6 * np.eye(n)
                
                # Solve K @ alpha = y_norm
                try:
                    alpha = np.linalg.solve(K, y_norm)
                except np.linalg.LinAlgError:
                    continue
                
                # Score: how well does the GP predict the training data?
                # (lower is better)
                pred = K @ alpha
                score = np.sum((pred - y_norm) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_length_scale = ls
            
            ls = best_length_scale
            sq_dists = np.sum(X_norm ** 2, axis=1)[:, None] + \
                       np.sum(X_norm ** 2, axis=1)[None, :] - \
                       2 * X_norm @ X_norm.T
            sq_dists = np.maximum(sq_dists, 0)
            K = np.exp(-sq_dists / (2 * ls ** 2))
            K += 1e-6 * np.eye(n)
            
            try:
                alpha = np.linalg.solve(K, y_norm)
            except np.linalg.LinAlgError:
                # Fallback: use the best training point
                return X_train[np.argmin(y_train)].copy()
            
            # Generate candidate points: mix of random and structured
            rng = np.random.default_rng(123)
            candidates = []
            
            # Random candidates in the box
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, d))
            
            # Structured candidates: perturbations of training points
            for i in range(min(n, n_candidates // 4)):
                for noise_scale in [0.3, 0.8, 1.5]:
                    cand = X_train[i] + rng.normal(0, noise_scale, d)
                    cand = np.clip(cand, -5, 5)
                    candidates.append(cand)
            
            # Center of the best region
            best_idx = np.argmin(y_train)
            for noise_scale in [0.2, 0.5, 1.0]:
                cand = X_train[best_idx] + rng.normal(0, noise_scale, d)
                cand = np.clip(cand, -5, 5)
                candidates.append(cand)
            
            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5, 5)
            
            # Evaluate GP posterior mean at candidates
            X_cand_norm = (candidates - X_min) / X_range
            X_cand_norm = np.clip(X_cand_norm, 0, 1)
            
            sq_dists_cand = np.sum(X_cand_norm ** 2, axis=1)[:, None] + \
                            np.sum(X_norm ** 2, axis=1)[None, :] - \
                            2 * X_cand_norm @ X_norm.T
            sq_dists_cand = np.maximum(sq_dists_cand, 0)
            K_cand = np.exp(-sq_dists_cand / (2 * ls ** 2))
            
            # Posterior mean: K_cand @ K^{-1} @ y
            try:
                mu = K_cand @ alpha  # in normalized y space
            except Exception:
                return X_train[np.argmin(y_train)].copy()
            
            # Convert back to original y scale
            mu_orig = mu * y_std + y_mean
            
            # Find the candidate with lowest predicted mean
            best_cand_idx = np.argmin(mu_orig)
            return candidates[best_cand_idx].copy()

        def surrogate_guided_start(n_samples=40):
            """
            Phase 1: Sample n_points to build a GP surrogate,
            then use it to identify the most promising basin center.
            Returns the recommended starting point.
            """
            rng = np.random.default_rng(7)
            
            # Sample points: mix of structured and random
            X_samples = []
            y_samples = []
            
            # Structured: valley points
            for a in [0.5, -0.5, 0.8, -0.8, 1.2, -1.2, 0.3, -0.3]:
                v = np.zeros(self.dim)
                v[0] = a
                for i in range(1, self.dim):
                    v[i] = a ** (2 ** i)
                v = np.clip(v, -5, 5)
                X_samples.append(v)
            
            # Center and corners
            X_samples.append(np.zeros(self.dim))
            X_samples.append(np.ones(self.dim))
            X_samples.append(-np.ones(self.dim))
            X_samples.append(np.ones(self.dim) * 3)
            X_samples.append(-np.ones(self.dim) * 3)
            
            # Random
            for _ in range(n_samples - len(X_samples)):
                X_samples.append(rng.uniform(-5, 5, self.dim))
            
            # Evaluate all samples
            for x in X_samples:
                if self.calls_made >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                x = np.clip(x, -5, 5)
                f = wrapped_func(x)
                y_samples.append(f)
            
            if len(y_samples) < 5:
                # Not enough samples, just return best found
                return self.best_x if self.best_x is not None else np.zeros(self.dim)
            
            X_samples = np.array(X_samples[:len(y_samples)])
            y_samples = np.array(y_samples)
            
            # Use GP to find the most promising point
            gp_start = gp_surrogate_minimize(X_samples, y_samples, n_candidates=30)
            
            return gp_start

        # === PHASE 1: Surrogate-assisted exploration ===
        # Use first ~40 evaluations to build a GP surrogate and find best basin
        surrogate_start = surrogate_guided_start(n_samples=40)
        
        if self.best_x is not None and surrogate_start is not None:
            # Compare: if GP suggests a different basin, verify it
            if not np.allclose(self.best_x, surrogate_start, atol=0.5):
                f_surr = wrapped_func(surrogate_start)
                if f_surr < self.best_f:
                    self.best_f = f_surr
                    self.best_x = surrogate_start.copy()
        
        # === PHASE 2: Multi-start L-BFGS-B from best basin ===
        rng = np.random.default_rng(42)
        
        # Build starts centered around the surrogate-identified basin
        center = self.best_x if self.best_x is not None else np.ones(self.dim)
        center = np.clip(center, -5, 5)
        
        starts = [center.copy()]
        
        # Perturbations of the center (local basin exploration)
        for noise_scale in [0.2, 0.5, 1.0, 2.0]:
            for _ in range(2):
                s = center + rng.normal(0, noise_scale, self.dim)
                s = np.clip(s, -5, 5)
                starts.append(s)
        
        # Additional structured starts (valley points)
        for a in [0.5, -0.5, 0.8, -0.8, 1.2, -1.2]:
            v = np.zeros(self.dim)
            v[0] = a
            for i in range(1, self.dim):
                v[i] = a ** (2 ** i)
            v = np.clip(v, -5, 5)
            starts.append(v)
        
        # A few random starts for diversity
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=0.1):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 12:
                continue
            
            # L-BFGS-B optimization from this start
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 60,
                        'maxls': 15,
                        'ftol': 1e-9,
                        'gtol': 1e-7,
                        'maxfun': 400
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                break
            except Exception:
                pass
            
            # Valley-following refinement
            if self.best_x is not None and self.calls_made < self.budget - 8:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=12)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    break
                except Exception:
                    pass
        
        # Final high-precision polish
        if self.best_x is not None and self.calls_made < self.budget - 5 and time.time() < self.deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxls': 20,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': 300
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
            except StopIteration:
                pass
            except Exception:
                pass
            
            # Final valley-following polish
            if self.best_x is not None and self.calls_made < self.budget - 3:
                try:
                    x_vf, f_vf = valley_follow(self.best_x, max_iters=8)
                    if f_vf < self.best_f:
                        self.best_f = f_vf
                        self.best_x = x_vf
                except StopIteration:
                    pass
                except Exception:
                    pass

        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)
        
        return self.best_x, self.best_f
