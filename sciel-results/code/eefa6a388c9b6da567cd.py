"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid optimizer for BBOB f4 (Büche-Rastrigin).
    
    Strategy:
    1. Phase 1: Sparse Gaussian Process (GP) surrogate with Expected Improvement (EI).
       - Uses first ~100-150 evaluations to build a surrogate model.
       - Sequentially selects points to maximize EI, balancing exploration/exploitation.
       - This is a structural change from pure stochastic search, aiming to capture
         the global basin structure and uncertainty of the rotated multimodal landscape.
    2. Phase 2: Local refinement using (1+1)-ES or Nelder-Mead on the best point found.
       - Uses remaining budget to polish the solution.
    
    Rationale:
    - Büche-Rastrigin is multimodal with a rotated landscape. Random restarts (current/inspiration 1)
      may waste budget on poor basins. GP surrogate can identify promising regions more efficiently
      by modeling the landscape and uncertainty.
    - EI acquisition function naturally balances exploration (high uncertainty) and exploitation
      (low predicted value).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        # --- Phase 1: GP Surrogate with EI ---
        
        # Parameters for GP
        # We use a simple RBF kernel with automatic relevance determination (ARD)
        # Hyperparameters: length_scales, noise, signal_variance
        # We will manually tune these or use a simple grid search if time permits,
        # but for speed, we start with reasonable defaults.
        
        # Initial points for GP: stratified random sampling
        # We want to cover the space [-5, 5]^dim
        # Let's use 50 initial points for the GP training set
        num_initial_gp = 50
        
        X = []
        Y = []
        
        # First point: origin (often good for symmetric functions, though f4 is shifted)
        x0 = np.zeros(self.dim)
        fx0 = evaluate(x0)
        if fx0 is None:
            return best_x, best_f
        X.append(x0)
        Y.append(fx0)
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()
            
        # Stratified random points
        # Divide each dimension into 3 bins: [-5, -1.67], [-1.67, 1.67], [1.67, 5]
        # Or simpler: random uniform in [-5, 5]
        for i in range(num_initial_gp - 1):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = evaluate(x)
            if fx is None:
                break
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        X = np.array(X)
        Y = np.array(Y)
        
        # GP Parameters
        # Standardize Y for better conditioning?
        # For simplicity, we'll work with raw Y but ensure numerical stability.
        # Length scales: start with 1.0 for all dimensions (isotropic)
        # Noise: small value to avoid singularity
        # Signal variance: variance of Y
        
        def gaussian_kernel(X1, X2, length_scales, signal_variance):
            """Compute RBF kernel matrix."""
            # X1: (N, D), X2: (M, D)
            # length_scales: (D,)
            # Returns: (N, M)
            
            # Compute squared distances
            # diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            # sq_dist = np.sum(diff ** 2 / length_scales[np.newaxis, np.newaxis, :] ** 2, axis=2)
            
            # More efficient:
            X1_scaled = X1 / length_scales
            X2_scaled = X2 / length_scales
            sq_dist = np.sum(X1_scaled ** 2, axis=1)[:, np.newaxis] + \
                      np.sum(X2_scaled ** 2, axis=1)[np.newaxis, :] - \
                      2 * np.dot(X1_scaled, X2_scaled.T)
            sq_dist = np.maximum(sq_dist, 0)  # Numerical stability
            
            return signal_variance * np.exp(-0.5 * sq_dist)
        
        # We will perform sequential EI for the remaining budget
        # Let's reserve the last 100 calls for local polishing
        local_search_budget = 100
        gp_max_calls = self.budget - local_search_budget
        
        # EI Parameters
        # We want to minimize, so EI = (y_best - mu) * Phi((y_best - mu)/sigma) + sigma * phi((y_best - mu)/sigma)
        # Where Phi and phi are CDF and PDF of standard normal
        
        def expected_improvement(mu, sigma, y_best):
            """
            mu: (N,) predicted mean
            sigma: (N,) predicted std
            y_best: float, current best
            """
            # Avoid division by zero
            sigma = np.maximum(sigma, 1e-8)
            z = (y_best - mu) / sigma
            # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
            # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
            from scipy.stats import norm
            ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei
        
        # Sequential EI loop
        while calls < gp_max_calls and time.time() < deadline:
            # 1. Update GP model
            # We need to compute the posterior mean and variance at candidate points.
            # To find the next point, we can either:
            # a) Sample many candidates and pick the one with highest EI
            # b) Use an optimization algorithm to maximize EI
            # For speed and simplicity, we'll use candidate sampling.
            
            # Generate candidate points
            num_candidates = 200
            # Mix of random points and points near current best
            X_cand = np.empty((num_candidates, self.dim))
            half = num_candidates // 2
            X_cand[:half] = rng.uniform(-5.0, 5.0, (half, self.dim))
            # Near best
            if best_x is not None:
                X_cand[half:] = best_x + rng.normal(0, 1.0, (num_candidates - half, self.dim))
                X_cand[half:] = np.clip(X_cand[half:], -5.0, 5.0)
            else:
                X_cand[half:] = rng.uniform(-5.0, 5.0, (num_candidates - half, self.dim))
                
            # 2. Compute GP predictions at candidates
            # Hyperparameters: length_scales, signal_variance, noise
            # We'll use fixed hyperparameters for now to avoid optimization overhead
            # If this performs poorly, we might need to optimize them.
            
            length_scales = np.ones(self.dim) * 1.5
            signal_variance = np.var(Y) if len(Y) > 1 else 1.0
            if signal_variance < 1e-6:
                signal_variance = 1.0
            noise = 1e-6
            
            # Compute kernel matrices
            K_train = gaussian_kernel(X, X, length_scales, signal_variance) + noise * np.eye(len(X))
            K_train_cand = gaussian_kernel(X, X_cand, length_scales, signal_variance)
            
            # Solve for weights
            # L = cholesky(K_train)
            # alpha = solve(L, Y)
            # mu = K_train_cand.T @ alpha
            # v = solve(L, K_train_cand)
            # var = signal_variance - sum(v**2, axis=0)
            
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L, Y)
                mu = K_train_cand.T @ alpha
                
                v = np.linalg.solve(L, K_train_cand)
                var = signal_variance - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-8)
                sigma = np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback: add jitter
                K_train += 1e-4 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L, Y)
                    mu = K_train_cand.T @ alpha
                    v = np.linalg.solve(L, K_train_cand)
                    var = signal_variance - np.sum(v ** 2, axis=0)
                    var = np.maximum(var, 1e-8)
                    sigma = np.sqrt(var)
                except Exception:
                    break # GP failed, break to local search
            
            # 3. Compute EI
            ei = expected_improvement(mu, sigma, best_f)
            
            # 4. Select best candidate
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            
            # 5. Evaluate
            fx_next = evaluate(x_next)
            if fx_next is None:
                break
            
            # 6. Update data
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx_next)
            
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next.copy()
                
            # Check time
            if time.time() > deadline:
                break

        # --- Phase 2: Local Refinement ---
        
        # Use remaining budget for local search
        if best_x is not None and calls < self.budget and time.time() < deadline:
            # Try (1+1)-ES first, then Nelder-Mead if time/budget allows
            
            sigma = 0.5
            sigma_min = 0.01
            sigma_max = 5.0
            stall_threshold = 20
            max_steps = self.budget - calls
            
            x = best_x.copy()
            fx = best_f
            
            for step in range(max_steps):
                if calls >= self.budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, sigma_max)
                    stall_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    stall_count += 1
                    if stall_count >= stall_threshold:
                        break
                        
            # If we still have budget and time, try Nelder-Mead polish
            if calls < self.budget and time.time() < deadline and best_x is not None:
                try:
                    def wrapped_func(x):
                        nonlocal calls
                        if calls >= self.budget:
                            return np.inf
                        x = np.clip(x, -5.0, 5.0)
                        fx = float(func(x))
                        calls += 1
                        return fx
                        
                    remaining = self.budget - calls
                    max_iter = min(50, max(10, remaining // 5))
                    
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6,
                                            'adaptive': True})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
