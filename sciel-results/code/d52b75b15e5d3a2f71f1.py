import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

class Optimizer:
    """Surrogate-assisted optimizer: GP exploration + L-BFGS-B refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        n = self.dim
        total_budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)

        # Storage for surrogate model
        X_history = []
        Y_history = []

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls >= total_budget:
                raise StopIteration("Budget exhausted")

            x = np.asarray(x, dtype=np.float64).flatten()
            if x.shape != (n,):
                x = x.reshape(n)
            if not np.all(np.isfinite(x)):
                return 1e10

            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()

            # Record for surrogate
            X_history.append(x_clipped.copy())
            Y_history.append(f)

            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Exploration for Surrogate
        # Allocate ~20% of budget for diverse sampling to build a GP model
        n_explore = max(20, int(total_budget * 0.2))
        
        # Generate diverse starting points
        starts = [np.zeros(n)]
        starts.append(np.ones(n) * 3.0)
        starts.append(np.ones(n) * -3.0)
        starts.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0]))
        starts.append(np.array([-1.0, 1.0, -1.0, 1.0, -1.0]))
        
        while len(starts) < n_explore:
            starts.append(rng.uniform(-4.5, 4.5, size=n))

        for x0 in starts:
            if calls >= total_budget or time.time() > deadline:
                break
            try:
                f0 = safe_func(x0)
            except (StopIteration, TimeoutError):
                break

        if not X_history:
            return best_x, best_f

        X_arr = np.array(X_history)
        Y_arr = np.array(Y_history)

        # Phase 2: Build Surrogate and Find Promising Basins
        gp = None
        try:
            # Standardize Y for better GP performance
            Y_mean = np.mean(Y_arr)
            Y_std = np.std(Y_arr)
            if Y_std < 1e-8:
                Y_std = 1.0
            Y_scaled = (Y_arr - Y_mean) / Y_std

            kernel = ConstantKernel(1.0) * Matern(length_scale=1.5, nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False, alpha=1e-10)
            gp.fit(X_arr, Y_scaled)
        except Exception:
            gp = None

        # Identify promising starting points for local optimization
        # Strategy: 
        # 1. The best found point
        # 2. Points where the GP predicts low values (exploitation)
        # 3. Points with high uncertainty (exploration) - though for Rosenbrock, exploitation is key
        
        candidate_starts = [best_x.copy()]
        
        if gp is not None:
            # Sample a grid or random points to evaluate GP mean and variance
            # We want points where GP mean is low.
            # Let's sample 50 random points and check GP predictions
            n_gp_samples = 50
            X_gp_samples = rng.uniform(-5, 5, size=(n_gp_samples, n))
            
            try:
                Y_pred, Y_std_pred = gp.predict(X_gp_samples, return_std=True)
                # Combine mean and variance: we want low mean. 
                # Acquisition function: negative mean (minimize mean)
                # Or Expected Improvement style. Let's just look at low mean points.
                # Sort by predicted mean
                indices = np.argsort(Y_pred)
                for idx in indices[:5]:
                    x_candidate = X_gp_samples[idx].copy()
                    # Check if it's far from existing best
                    if np.linalg.norm(x_candidate - best_x) > 0.5:
                        candidate_starts.append(x_candidate)
            except Exception:
                pass

        # Also add a few random points from the exploration phase that were good
        # Sort history by f-value
        history_sorted = sorted(zip(Y_arr, X_arr), key=lambda item: item[0])
        for f_val, x_val in history_sorted[:3]:
            if x_val is not None and np.linalg.norm(x_val - best_x) > 0.5:
                candidate_starts.append(x_val.copy())

        # Deduplicate candidate starts
        unique_starts = []
        for x in candidate_starts:
            is_unique = True
            for u in unique_starts:
                if np.linalg.norm(x - u) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                unique_starts.append(x)

        # Phase 3: Local Refinement with L-BFGS-B
        # Use the remaining budget for local optimization from the best candidates
        # L-BFGS-B is excellent for Rosenbrock once in the valley
        
        for x_start in unique_starts:
            if calls >= total_budget or time.time() > deadline:
                break
            
            # Skip if this start is worse than current best by a lot? 
            # No, L-BFGS-B might find a better basin.
            
            try:
                res = minimize(
                    safe_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-14,
                        'gtol': 1e-8,
                        'disp': False,
                        'maxls': 20
                    }
                )
                # safe_func updates best_x and best_f internally
            except (StopIteration, TimeoutError):
                # Budget or time exceeded, stop all further optimizations
                break
            except Exception:
                # If L-BFGS-B fails for other reasons, continue to next candidate
                continue

        return best_x, best_f
