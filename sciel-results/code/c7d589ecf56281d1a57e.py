import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import t as student_t

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None
        self.X_data = []
        self.y_data = []
        self.gp_ls = None  # adaptive length scales

    def __call__(self, func):
        self.deadline = time.time() + 285
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf
        self.X_data = []
        self.y_data = []

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            self.X_data.append(x.copy())
            self.y_data.append(f)
            return f

        # Phase 1: Initial sampling with diverse starts
        initial_points = [np.zeros(self.dim)]
        # Add axis-aligned points
        for axis in range(self.dim):
            for sign in [-1, 1]:
                p = np.zeros(self.dim)
                p[axis] = sign * 2.0
                initial_points.append(p)
        # Add random points
        for _ in range(6):
            initial_points.append(np.random.uniform(-4, 4, self.dim))
        
        for x0 in initial_points:
            if self.calls >= 25 or time.time() > self.deadline - 30:
                break
            wrapped_func(x0)
        
        # Phase 2: Bayesian Optimization with GP surrogate and Student-t EI
        bo_budget = self.budget - self.calls
        if bo_budget > 20 and time.time() < self.deadline - 20:
            self._bayesian_optimize(wrapped_func, bo_budget)
        
        # Phase 3: Final polish - first Nelder-Mead, then L-BFGS-B
        if self.calls < self.budget and time.time() < self.deadline - 15:
            try:
                polish_budget = self.budget - self.calls
                maxiter = max(1, polish_budget // 4)
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-12,
                                        'initial_simplex': self._make_simplex(self.best_x)})
                if self.best_f > res.fun:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 4: L-BFGS-B polish if budget remains
        if self.calls < self.budget and time.time() < self.deadline - 10:
            try:
                remaining = self.budget - self.calls
                if remaining >= 5:
                    res = minimize(wrapped_func, self.best_x, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * self.dim,
                                   options={'maxiter': max(1, remaining // 2), 'gtol': 1e-8})
                    if self.best_f > res.fun:
                        self.best_f = res.fun
                        self.best_x = res.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f

    def _make_simplex(self, x0):
        """Create a simplex around x0 for the polish pass."""
        simplex = [x0.copy()]
        for i in range(self.dim):
            v = x0.copy()
            step = 0.1
            v[i] += step
            simplex.append(v)
        return np.array(simplex)

    def _gp_predict(self, X_test):
        """Gaussian Process prediction using RBF kernel with adaptive length scales."""
        X_train = np.array(self.X_data)
        y_train = np.array(self.y_data)
        
        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        # RBF kernel with vectorized distance computation
        def rbf_kernel(X1, X2, length_scales, variance):
            # Compute squared distances efficiently
            sq_dist = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            # Scale by length scales
            sq_dist /= length_scales[np.newaxis, np.newaxis, :] ** 2
            sq_dist = np.sum(sq_dist, axis=2)
            K = variance * np.exp(-0.5 * sq_dist)
            return K
        
        # Adaptive hyperparameters based on data spread
        if self.gp_ls is None:
            # Estimate length scales from data spread
            data_range = np.ptp(X_train, axis=0) + 1e-6
            self.gp_ls = data_range * 0.5
        else:
            # Gradually adapt length scales toward data spread
            data_range = np.ptp(X_train, axis=0) + 1e-6
            target_ls = data_range * 0.5
            self.gp_ls = 0.8 * self.gp_ls + 0.2 * target_ls
        
        length_scales = np.clip(self.gp_ls, 0.1, 5.0)
        variance = 1.0
        noise = 1e-3
        
        # Train kernel matrix
        K = rbf_kernel(X_train, X_train, length_scales, variance)
        K += noise * np.eye(len(X_train))
        
        # Test kernel matrix
        K_star = rbf_kernel(X_test, X_train, length_scales, variance)
        
        # Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-5 * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback: return uniform prediction
                n_test = len(X_test)
                mean = np.full(n_test, y_mean)
                var = np.full(n_test, y_std ** 2)
                return mean, var
        
        # Solve for alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        # Predict mean and variance
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = variance - np.sum(v * K_star, axis=1)
        var = np.maximum(var, 1e-8)
        
        # Denormalize
        mean = mean * y_std + y_mean
        var = var * y_std ** 2
        
        return mean, var

    def _student_t_ei(self, mu, sigma, f_best, nu=3):
        """Student-t Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-8)
        improvement = f_best - mu
        z = improvement / sigma
        # Student-t CDF and PDF
        t_cdf = student_t.cdf(z, df=nu)
        t_pdf = student_t.pdf(z, df=nu)
        ei = improvement * t_cdf + sigma * t_pdf
        return ei

    def _bayesian_optimize(self, eval_func, budget):
        """Bayesian Optimization loop with GP and Student-t EI."""
        # Use Sobol-like sampling for better coverage
        max_iter = min(budget, 150)
        
        for it in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline - 15:
                break
            
            # Generate candidate points using stratified sampling
            n_candidates = 500
            # Mix of random and Latin-hypercube-style points
            candidates = np.random.uniform(-5, 5, (n_candidates, self.dim))
            
            # Add points near current best for exploitation
            n_exploit = 50
            exploit_pts = self.best_x + np.random.normal(0, 0.5, (n_exploit, self.dim))
            exploit_pts = np.clip(exploit_pts, -5, 5)
            candidates = np.vstack([candidates, exploit_pts])
            
            # Predict mean and variance
            mu, var = self._gp_predict(candidates)
            sigma = np.sqrt(var)
            
            # Compute EI with exploration/exploitation balance
            xi = 0.01  # small exploitation parameter
            ei = self._student_t_ei(mu, sigma, self.best_f - xi, nu=3)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = eval_func(x_next)
            
            # If we got a valid result, continue
            if np.isinf(f_next):
                break
