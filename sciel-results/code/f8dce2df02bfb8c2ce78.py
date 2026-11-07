"""Gaussian Process surrogate-guided search with L-BFGS-B refinement for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Global surrogate-guided exploration with local refinement for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        from scipy.optimize import minimize
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
        from sklearn.utils.parallel import delayed
        from joblib import Parallel

        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0

        # Storage for surrogate model
        X_samples = []
        y_samples = []

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            call_count += 1
            X_samples.append(x_clipped.copy())
            y_samples.append(f)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Initial sampling to build surrogate
        # Sample a structured set of points to get initial coverage
        n_initial = min(40, budget // 5)
        
        # Start with origin
        evaluate(np.zeros(dim))
        
        # Add random points
        for _ in range(n_initial - 1):
            if call_count >= budget or not time_ok():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        # Phase 2: Surrogate-guided exploration using Expected Improvement
        # Build initial GP
        X_arr = np.array(X_samples)
        y_arr = np.array(y_samples)
        
        # Normalize y for GP stability
        y_mean = y_arr.mean()
        y_std = max(y_arr.std(), 1e-8)
        y_norm = (y_arr - y_mean) / y_std

        # Define kernel: Matern with ARD + noise
        length_scale = 1.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=1e-6)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10)
        gp.fit(X_arr, y_norm)

        # Explore using Expected Improvement (EI)
        n_surrogate_steps = min(budget - call_count - 50, 150)  # Leave budget for local refinement
        
        for step in range(n_surrogate_steps):
            if call_count >= budget - 50 or not time_ok():
                break
            
            # Generate candidate points for EI maximization
            n_candidates = 500
            candidates = rng.uniform(-5.0, 5.0, (n_candidates, dim))
            
            # Predict mean and std at candidates
            mu, sigma = gp.predict(candidates, return_std=True)
            
            # Expected Improvement for minimization
            # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
            # Use normalized values
            best_norm = (best_f - y_mean) / y_std
            
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (best_norm - mu) / sigma
                sigma_safe = np.where(sigma > 1e-10, sigma, 1.0)
                z_safe = np.where(sigma > 1e-10, z, 0.0)
                
                from scipy.stats import norm
                ei = np.where(sigma > 1e-10, 
                              (best_norm - mu) * norm.cdf(z_safe) + sigma_safe * norm.pdf(z_safe), 
                              0.0)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            f_next = evaluate(x_next)
            
            # Update GP with new point
            if not time_ok():
                break
                
            X_arr = np.vstack([X_arr, x_next.reshape(1, -1)])
            y_new_norm = (f_next - y_mean) / y_std
            y_arr = np.append(y_arr, f_next)
            y_norm = np.append(y_norm, y_new_norm)
            
            # Refit GP periodically (every 5 points) for efficiency
            if len(X_arr) % 5 == 0:
                try:
                    gp.fit(X_arr, y_norm)
                except Exception:
                    pass

        # Phase 3: Local refinement with L-BFGS-B
        local_budget = budget - call_count
        if local_budget > 10 and time_ok():
            def wrapper(x):
                if not time_ok() or call_count >= budget:
                    return float('inf')
                return evaluate(x)

            # Try L-BFGS-B from current best
            try:
                res = minimize(wrapper, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxls': 20})
            except Exception:
                pass

            # Try from a few perturbed points
            if call_count < budget - 5 and time_ok():
                for attempt in range(3):
                    if call_count >= budget - 5 or not time_ok():
                        break
                    x0 = best_x + rng.normal(0, 0.2, dim)
                    x0 = np.clip(x0, -5.0, 5.0)
                    try:
                        res2 = minimize(wrapper, x0, method='L-BFGS-B',
                                        bounds=[(-5.0, 5.0)] * dim,
                                        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10,
                                                 'maxls': 20})
                    except Exception:
                        pass

        return best_x, best_f

# EVOLVE-BLOCK-END
