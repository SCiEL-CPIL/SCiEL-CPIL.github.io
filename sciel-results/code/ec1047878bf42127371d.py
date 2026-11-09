"""GP-based Bayesian Optimization with L-BFGS-B polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate for multimodal Weierstrass.
    Uses EI acquisition with adaptive length scales, initial space-filling exploration,
    and L-BFGS-B local refinement on the best candidates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        rng = np.random.default_rng(42)

        # --- Gaussian Process Surrogate ---
        class GPSurrogate:
            def __init__(self, dim, length_scale=1.0, variance=1.0, noise=1e-6):
                self.dim = dim
                self.length_scale = length_scale
                self.variance = variance
                self.noise = noise
                self.X = None
                self.y = None
                self.L = None
                self.alpha = None

            def _kern(self, X1, X2):
                """RBF kernel with adaptive length scale."""
                ls = self.length_scale
                sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sqdist = np.maximum(sqdist, 0)
                return self.variance * np.exp(-0.5 * sqdist / (ls * ls))

            def fit(self, X, y):
                self.X = np.asarray(X, dtype=float)
                self.y = np.asarray(y, dtype=float)
                K = self._kern(self.X, self.X)
                K += self.noise * np.eye(len(self.X))
                try:
                    self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(self.X)))
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
                except np.linalg.LinAlgError:
                    K += 1e-8 * np.eye(len(self.X))
                    self.L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

            def predict(self, X_new):
                X_new = np.asarray(X_new, dtype=float)
                K_s = self._kern(self.X, X_new)
                K_ss = self._kern(X_new, X_new)
                v = np.linalg.solve(self.L, K_s.T)
                mu = v.T @ self.alpha
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)

        def ei_next_point(gp, n_candidates=50, exploit_explore=1.0):
            """Select next point using Expected Improvement acquisition."""
            if best_f == np.inf:
                return rng.uniform(-5, 5, dim)
            
            mu, sigma = gp.predict(rng.uniform(-5, 5, dim * n_candidates).reshape(-1, dim))
            # Add some points near the current best
            near_best = best_x + rng.normal(0, 0.5, (n_candidates // 2, dim))
            near_best = np.clip(near_best, -5, 5)
            mu2, sigma2 = gp.predict(near_best)
            
            all_mu = np.concatenate([mu, mu2])
            all_sigma = np.concatenate([sigma, sigma2])
            all_x = np.vstack([rng.uniform(-5, 5, (n_candidates, dim)), near_best])
            
            # EI for minimization
            improvement = best_f - all_mu
            z = improvement / (all_sigma + 1e-10)
            ei = improvement * norm.cdf(z) + all_sigma * norm.pdf(z)
            ei *= exploit_explore
            
            idx = np.argmax(ei)
            return all_x[idx]

        # --- Phase 1: Initial space-filling exploration ---
        n_init = min(20, budget)
        X_init = rng.uniform(-5, 5, (n_init, dim))
        y_init = []
        for i in range(n_init):
            if n_calls >= budget or check_time():
                break
            val, x_eval = call_func(X_init[i])
            if val is not None:
                y_init.append(val)
                update_best(x_eval, val)
        
        if best_x is None:
            return None

        # --- Phase 2: GP-guided Bayesian Optimization ---
        gp = GPSurrogate(dim, length_scale=2.0, variance=1.0, noise=1e-5)
        
        X_obs = []
        y_obs = []
        for i in range(n_init):
            if i < len(y_init):
                X_obs.append(X_init[i])
                y_obs.append(y_init[i])
        
        if len(X_obs) >= 3:
            gp.fit(np.array(X_obs), np.array(y_obs))
        
        # Adaptive BO loop
        exploit_explore = 1.0
        last_improvement_call = n_calls
        stagnation_count = 0
        max_stagnation = 30
        
        while n_calls < budget and not check_time():
            # Select next point
            x_next = ei_next_point(gp, n_candidates=40, exploit_explore=exploit_explore)
            
            # Evaluate
            val, x_eval = call_func(x_next)
            if val is None:
                break
            
            improved = update_best(x_eval, val)
            X_obs.append(x_eval)
            y_obs.append(val)
            
            if improved:
                last_improvement_call = n_calls
                stagnation_count = 0
                # Reduce exploration when improving
                exploit_explore = max(0.5, exploit_explore * 0.95)
            else:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    # Reset exploration
                    exploit_explore = 1.5
                    stagnation_count = 0
                    # Try a random restart
                    x_rand = rng.uniform(-5, 5, dim)
                    val_rand, x_rand_eval = call_func(x_rand)
                    if val_rand is not None:
                        update_best(x_rand_eval, val_rand)
                        X_obs.append(x_rand_eval)
                        y_obs.append(val_rand)
            
            # Update GP periodically (every 5 new points or when stagnating)
            if len(X_obs) % 5 == 0 or stagnation_count > 10:
                # Adaptive length scale: shrink when close to optimum
                if best_f < 10:
                    gp.length_scale = max(0.3, gp.length_scale * 0.9)
                gp.fit(np.array(X_obs), np.array(y_obs))
            
            # Time check
            if check_time():
                break

        # --- Phase 3: L-BFGS-B local refinement ---
        if n_calls < budget and not check_time() and best_x is not None:
            remaining = budget - n_calls
            max_l_bfgs_calls = min(remaining, 50)
            if max_l_bfgs_calls > 5:
                calls_used = 0
                x_best_local = best_x.copy()
                f_best_local = best_f
                
                def objective(x):
                    nonlocal calls_used
                    if calls_used >= max_l_bfgs_calls or n_calls >= budget or check_time():
                        return f_best_local
                    val, x_eval = call_func(x)
                    if val is None:
                        return f_best_local
                    calls_used += 1
                    return val
                
                try:
                    result = minimize(
                        objective,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': max_l_bfgs_calls, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if result.fun < f_best_local:
                        x_best_local = np.clip(result.x, -5, 5)
                        f_best_local = result.fun
                        update_best(x_best_local, f_best_local)
                except Exception:
                    pass

        if best_x is None:
            return None
        return best_x, best_f

# EVOLVE-BLOCK-END
