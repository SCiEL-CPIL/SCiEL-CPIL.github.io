import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290  # 10s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Store all observations for surrogate model
        X_obs = []
        y_obs = []

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if np.isnan(f_val) or np.isinf(f_val):
                f_val = best_f if best_f != float('inf') else 0.0

            X_obs.append(np.array(x))
            y_obs.append(f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Initial Multi-Start Global Search
        # Use a mix of deterministic and random points to quickly find good basins
        num_starts = 15
        start_points = []
        # Include origin and a few deterministic points
        start_points.append(np.zeros(N))
        start_points.append(np.ones(N) * 2.0)
        start_points.append(np.ones(N) * -2.0)
        start_points.append(np.ones(N) * 4.0)
        start_points.append(np.ones(N) * -4.0)
        start_points.append(np.ones(N) * 1.0)
        start_points.append(np.ones(N) * -1.0)
        
        for _ in range(num_starts - len(start_points)):
            start_points.append(rng.uniform(-4.5, 4.5, size=N))

        # Evaluate starts and sort
        start_results = []
        try:
            for x in start_points:
                f_val = evaluate(x)
                start_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        if not start_results:
            return best_x, best_f

        start_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization with L-BFGS-B
        # Run L-BFGS-B from the best starting points to quickly converge to local minima
        def lbf_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if np.isnan(f_val) or np.isinf(f_val):
                f_val = best_f if best_f != float('inf') else 0.0

            X_obs.append(np.array(x))
            y_obs.append(f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Allocate budget for local searches
        # We have 500 calls. ~15 used for starts.
        # Let's try 5 local searches.
        local_starts = [x for _, x in start_results[:5]]
        
        for i, start_x in enumerate(local_starts):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # Limit iterations to ensure we finish multiple starts
            max_iter = 150 if i == 0 else 80
            
            try:
                res = minimize(
                    lbf_wrapper, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        # Phase 3: Surrogate-Assisted Global Refinement
        # Use a Gaussian Process to guide exploration in unexplored regions
        # and exploitation near the current best
        if calls_made < budget and time.time() - start_time < time_limit and len(X_obs) >= 10:
            try:
                X_train = np.array(X_obs)
                y_train = np.array(y_obs)
                
                # Normalize inputs to [0, 1] for better GP performance
                X_min = X_train.min(axis=0)
                X_max = X_train.max(axis=0)
                X_range = X_max - X_min
                X_range[X_range == 0] = 1.0
                
                # Normalize outputs
                y_mean = y_train.mean()
                y_std = y_train.std()
                if y_std == 0:
                    y_std = 1.0
                y_norm = (y_train - y_mean) / y_std
                
                # Fit GP model
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(N), nu=2.5) + WhiteKernel(noise_level=1e-10)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5, normalize_y=True)
                gp.fit(X_train, y_train)
                
                # Use Expected Improvement (EI) for acquisition
                def expected_improvement(X, y_best):
                    y_pred, y_std = gp.predict(X, return_std=True)
                    y_std = np.maximum(y_std, 1e-10)
                    z = (y_pred - y_best) / y_std
                    from scipy.stats import norm
                    ei = (y_pred - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
                    return ei.mean()
                
                # Generate candidate points for EI evaluation
                num_candidates = 200
                candidates = []
                
                # Mix of random points and points near current best
                for _ in range(num_candidates):
                    if rng.random() < 0.3:
                        # Exploit: near current best
                        x = best_x + rng.normal(0, 0.5, size=N)
                        x = np.clip(x, -5, 5)
                    else:
                        # Explore: random in the box
                        x = rng.uniform(-5, 5, size=N)
                    candidates.append(x)
                
                candidates = np.array(candidates)
                
                # Evaluate EI for all candidates
                y_best_norm = (best_f - y_mean) / y_std
                ei_values = []
                for i in range(0, len(candidates), 50):
                    batch = candidates[i:i+50]
                    y_pred, y_std = gp.predict(batch, return_std=True)
                    y_std = np.maximum(y_std, 1e-10)
                    z = (y_pred - y_best_norm) / y_std
                    from scipy.stats import norm
                    ei_batch = (y_pred - y_best_norm) * norm.cdf(z) + y_std * norm.pdf(z)
                    ei_values.extend(ei_batch)
                
                ei_values = np.array(ei_values)
                
                # Select top 5 candidates by EI
                top_indices = np.argsort(ei_values)[::-1][:5]
                
                # Evaluate these candidates
                for idx in top_indices:
                    if calls_made >= budget or time.time() - start_time > time_limit:
                        break
                    x = candidates[idx]
                    f_val = evaluate(x)
                    
                    # If we found a better point, do a quick local refinement
                    if f_val < best_f:
                        try:
                            res = minimize(
                                lbf_wrapper, 
                                x, 
                                method='L-BFGS-B', 
                                bounds=[(-5, 5)] * N,
                                options={
                                    'maxiter': 50,
                                    'ftol': 1e-10,
                                    'gtol': 1e-8,
                                    'maxls': 15
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = np.array(res.x)
                        except (TimeoutError, RuntimeError):
                            break
                        except Exception:
                            pass
                            
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead Polish
        # Nelder-Mead can refine the solution further, especially near the optimum
        if calls_made < budget and time.time() - start_time < time_limit and best_x is not None:
            def nm_wrapper(x):
                nonlocal calls_made, best_x, best_f
                if time.time() - start_time > time_limit:
                    raise TimeoutError()
                if calls_made >= self.budget:
                    raise RuntimeError()
                
                calls_made += 1
                try:
                    f_val = func(x)
                except Exception:
                    return best_f if best_f != float('inf') else 0.0
                
                if np.isnan(f_val) or np.isinf(f_val):
                    f_val = best_f if best_f != float('inf') else 0.0

                X_obs.append(np.array(x))
                y_obs.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)
                return f_val

            try:
                res = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
