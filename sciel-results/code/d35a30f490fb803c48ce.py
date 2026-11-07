import numpy as np
import time
import scipy.optimize as opt
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration

            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                if len(x) > dim:
                    x = x[:dim]
                else:
                    x_new = np.zeros(dim)
                    x_new[:len(x)] = x
                    x = x_new

            x = np.clip(x, -5.0, 5.0)

            try:
                val = float(func(x))
            except Exception:
                val = np.inf

            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Data storage for Bayesian Optimization
        X_obs = []
        Y_obs = []

        def record(x, y):
            X_obs.append(np.asarray(x, dtype=float).copy())
            Y_obs.append(float(y))

        # Helper: Halton sequence for space-filling initial points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2) # Bases 2, 3, 5, 7, 11...
            return points

        # --- Phase 1: Initial Diverse Sampling ---
        # Use a mix of Halton (structured) and random points to cover the space.
        # Katsuura is multimodal, so good initial coverage is key.
        n_halton = 20
        n_random = 20
        rng = np.random.default_rng(42)

        initial_points = halton_sequence(n_halton, dim) * 10.0 - 5.0 # Scale to [-5, 5]
        random_points = rng.uniform(-5.0, 5.0, size=(n_random, dim))

        # Add a few structured points (center, corners)
        structured_points = [
            np.zeros(dim),
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0
        ]

        all_initial = np.vstack([initial_points, random_points, np.array(structured_points)])

        for x0 in all_initial:
            if calls >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            try:
                val = safe_func(x0)
                record(x0, val)
            except (TimeoutError, StopIteration):
                break
            except Exception:
                continue

        if best_x is None:
            # Fallback if no calls were made
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
                record(best_x, best_f)
            except Exception:
                best_f = func(best_x)
                calls += 1
                record(best_x, best_f)

        # --- Phase 2: Bayesian Optimization with GP Surrogate ---
        # Use GP to model the function and EI to select next points.
        # This is efficient for expensive black-box functions with limited budget.
        
        # GP Parameters
        length_scale = 1.0
        signal_variance = 1.0
        noise_variance = 1e-6
        xi = 0.01 # Exploration constant for EI

        def kernel(X1, X2):
            # Matern 5/2 kernel is good for non-smooth functions like Katsuura
            # But RBF is simpler and often sufficient. Let's try Matern 5/2 for robustness.
            d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
            r = np.sqrt(np.maximum(d2, 1e-12)) / length_scale
            return signal_variance * (1 + np.sqrt(5) * r + 5/3 * r**2) * np.exp(-np.sqrt(5) * r)

        def gp_predict(X_new):
            if len(X_obs) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))
            
            X_train = np.array(X_obs)
            Y_train = np.array(Y_obs)
            
            K = kernel(X_train, X_train)
            K += noise_variance * np.eye(len(X_train))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-8 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return np.mean(Y_train), np.std(Y_train)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            K_s = kernel(X_train, X_new)
            K_ss = kernel(X_new, X_new)
            
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-12)
            
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_y, xi=0.01):
            z = (mu - best_y - xi) / sigma
            ei = (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # BO Loop
        # We will run BO until budget or time is exhausted.
        # To improve convergence, we can periodically restart NM from the best BO point.
        
        while calls < budget and time.time() < deadline:
            # Generate candidate points for BO
            # Use a mix of random and Latin Hypercube sampling for candidates
            n_candidates = 50
            candidates = rng.uniform(-5.0, 5.0, size=(n_candidates, dim))
            
            # Also add some points near the current best
            if best_x is not None:
                local_candidates = best_x + rng.normal(0, 0.5, size=(20, dim))
                local_candidates = np.clip(local_candidates, -5.0, 5.0)
                candidates = np.vstack([candidates, local_candidates])
            
            # Predict mean and variance for candidates
            try:
                mu, sigma = gp_predict(candidates)
                ei = expected_improvement(mu, sigma, best_f, xi)
                
                # Select the point with max EI
                idx = np.argmax(ei)
                x_next = candidates[idx]
                
            except Exception:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5.0, 5.0, size=dim)

            # Evaluate the selected point
            try:
                val = safe_func(x_next)
                record(x_next, val)
            except (TimeoutError, StopIteration):
                break
            except Exception:
                continue

            # Periodic Local Refinement
            # Every 20 calls, run a short Nelder-Mead from the best point to refine
            if calls % 20 == 0 and calls < budget - 10 and time.time() < deadline:
                try:
                    res = opt.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 5,
                            'xatol': 1e-4,
                            'fatol': 1e-4
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                        record(best_x, best_f)
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    pass

        # --- Phase 3: Final Polish ---
        if best_x is not None and calls < budget and time.time() < deadline:
            try:
                remaining = budget - calls
                if remaining >= 5:
                    est_maxiter_p3 = max(1, remaining // (dim + 1))
                    
                    res = opt.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': est_maxiter_p3,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except Exception:
                best_f = func(best_x)

        return best_x, best_f
