import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """
    Surrogate-Assisted Bayesian Optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    1. Initial Exploration: Use a small number of random or Sobol-like points to seed the model.
    2. Surrogate Modeling: Fit a Gaussian Process (GP) to the observed data.
    3. Acquisition: Use Expected Improvement (EI) to select the next point, balancing exploration and exploitation.
    4. Local Refinement: When the GP predicts high confidence in a local minimum, use a fast (1+1)-ES or L-BFGS-B to exploit it.
    
    This approach is structurally different from CMA-ES as it builds an explicit probabilistic model of the landscape,
    which is particularly effective for expensive or complex functions where direct population sampling might waste
    evaluations in flat regions or outside the basin.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._np_random = np.random.default_rng(42)

    def _fit_gp(self, X, Y, theta=None):
        """
        Fit a simple GP model.
        Returns: mean, std, and the optimal theta (length scales and variance).
        """
        n = X.shape[0]
        d = X.shape[1]
        
        if n == 1:
            return np.array([Y[0]]), np.array([1.0]), np.ones(d) * 1.0, 1.0

        # Normalize inputs and outputs for numerical stability
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X_mean) / X_std
        
        Y_mean = Y.mean()
        Y_std = Y.std()
        Y_std = max(Y_std, 1e-8)
        Y_norm = (Y - Y_mean) / Y_std

        # RBF Kernel
        def kernel(X1, X2, theta):
            # theta: [length_scales..., noise]
            ls = theta[:-1]
            noise = theta[-1]
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2 / (ls ** 2), axis=2)
            K = np.exp(-0.5 * sq_dist)
            if np.isclose(X1.shape[0], X2.shape[0]):
                K[np.diag_indices_from(K)] += noise
            return K

        # Initialize theta
        if theta is None:
            theta = np.concatenate([np.ones(d), [1e-6]])
        
        # Simple optimization of log-marginal-likelihood
        def neg_log_mll(theta):
            try:
                K = kernel(X_norm, X_norm, theta)
                L = np.linalg.cholesky(K + 1e-9 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                logdet = 2 * np.sum(np.log(np.diag(L)))
                mll = -0.5 * Y_norm.T @ alpha - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
                return -mll
            except:
                return 1e6

        # Optimize theta using L-BFGS-B
        try:
            res = minimize(neg_log_mll, theta, method='L-BFGS-B', bounds=[(1e-3, 10)] * d + [(1e-9, 1e-2)])
            theta_opt = res.x
        except:
            theta_opt = theta

        K = kernel(X_norm, X_norm, theta_opt)
        L = np.linalg.cholesky(K + 1e-9 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))

        # Predict mean and std at test points
        def predict(X_test):
            X_test_norm = (X_test - X_mean) / X_std
            K_s = kernel(X_test_norm, X_norm, theta_opt)
            mu = K_s @ alpha * Y_std + Y_mean
            v = np.linalg.solve(L, K_s.T)
            var = 1.0 - np.sum(v ** 2, axis=1) # Assuming unit variance for Y_norm
            var = np.maximum(var, 0)
            std = np.sqrt(var) * Y_std
            return mu, std

        return predict, theta_opt

    def _expected_improvement(self, mu, std, best_f, xi=0.01):
        """
        Calculate Expected Improvement.
        """
        best_f_adj = best_f - xi
        z = (best_f_adj - mu) / (std + 1e-10)
        ei = (best_f_adj - mu) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def _local_refinement(self, func, x0, max_calls, deadline, rng):
        """
        (1+1)-ES for local refinement.
        """
        n = self.dim
        x = x0.copy()
        fx = func(x)
        calls_made = 1
        sigma = 0.5
        best_x = x.copy()
        best_f = fx
        
        while calls_made < max_calls and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.05
                best_x = x.copy()
                best_f = fx
            else:
                sigma *= 0.95
                if sigma < 1e-6:
                    break
                    
        return best_x, best_f, calls_made

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        
        # Data storage
        X = []
        Y = []
        best_x = np.zeros(n)
        best_f = float('inf')
        
        rng = np.random.default_rng(123)
        
        # Phase 1: Initial Exploration
        # Use a small number of diverse points to seed the GP
        n_initial = 10
        for _ in range(n_initial):
            if calls_made >= total_budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, n)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if not X:
            return best_x, best_f

        X = np.array(X)
        Y = np.array(Y)
        
        # Phase 2: Bayesian Optimization Loop
        # We use a small local search budget for refinement
        local_refinement_budget = 50
        
        while calls_made < total_budget and time.time() < deadline:
            # Fit GP
            if len(X) < 2:
                # If we have very few points, just sample randomly
                x = rng.uniform(-5, 5, n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                X = np.vstack([X, x]) if isinstance(X, np.ndarray) else np.array([x])
                Y = np.append(Y, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                continue
            
            try:
                predict, _ = self._fit_gp(X, Y)
            except:
                # If GP fitting fails, fall back to random sampling
                x = rng.uniform(-5, 5, n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                X = np.vstack([X, x]) if isinstance(X, np.ndarray) else np.array([x])
                Y = np.append(Y, fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                continue

            # Sample candidate points for acquisition
            n_candidates = 100
            candidates = rng.uniform(-5, 5, (n_candidates, n))
            
            # Evaluate acquisition function
            mu, std = predict(candidates)
            ei = self._expected_improvement(mu, std, best_f)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            fx = func(x_next)
            calls_made += 1
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x_next.copy()
                
                # If we found a new best, consider local refinement
                # Only if we have enough budget left
                remaining = total_budget - calls_made
                if remaining > local_refinement_budget and time.time() < deadline - 10:
                    # Run local refinement from this point
                    # But we must be careful not to exceed budget
                    # We allocate a small portion of remaining budget
                    alloc = min(remaining // 2, local_refinement_budget)
                    if alloc > 5:
                        x_ref, f_ref, c_used = self._local_refinement(
                            func, best_x, alloc, deadline, rng
                        )
                        calls_made += c_used
                        # Add refined points to history? 
                        # For simplicity, we just update best and continue BO loop
                        # But adding the final point to X,Y helps the model
                        X = np.vstack([X, x_ref])
                        Y = np.append(Y, f_ref)
                        
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()
                            
        return best_x, best_f
