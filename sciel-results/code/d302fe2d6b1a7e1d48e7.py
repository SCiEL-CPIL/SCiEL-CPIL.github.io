import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """
    Optimizer using Gaussian Process Surrogate-Assisted Bayesian Optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        # Data storage
        X = []
        Y = []

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            
            if time.time() > deadline:
                return float('inf')
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # GP Parameters
        # We use a fixed RBF kernel with lengthscale 1.0 and signal variance 1.0
        # This is a heuristic but robust for normalized-ish spaces like [-5,5]^5
        lengthscale = 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def kernel(X1, X2):
            # RBF Kernel
            # X1: (n, d), X2: (m, d)
            # dist^2 = sum(x1^2) - 2 x1.x2 + sum(x2^2)
            X1_sq = np.sum(X1**2, axis=1)
            X2_sq = np.sum(X2**2, axis=1)
            dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
            dist_sq = np.maximum(dist_sq, 0)
            return signal_var * np.exp(-0.5 * dist_sq / (lengthscale**2))

        def fit_gp(X, Y):
            n = X.shape[0]
            K = kernel(X, X)
            K += noise_var * np.eye(n)
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                L = np.linalg.pinv(K)
                alpha = np.linalg.solve(K, Y)
            return L, alpha

        def gp_predict(X_new, L, alpha, X_train, Y_train):
            # X_new: (m, d)
            # Returns mean, var
            K_s = kernel(X_new, X_train)
            K_ss = kernel(X_new, X_new)
            
            # Mean
            mu = K_s @ alpha
            
            # Variance
            # var = K_ss - K_s @ K^-1 @ K_s.T
            # Using Cholesky: K^-1 = L^-T L^-1
            # If L is cholesky factor: K = L L^T
            # K^-1 = L^-T L^-1
            # K_s @ K^-1 = K_s @ L^-T @ L^-1
            # Let V = L^-1 @ K_s.T
            # Then K_s @ K^-1 @ K_s.T = V.T @ V
            try:
                V = np.linalg.solve(L.T, K_s.T)
            except np.linalg.LinAlgError:
                V = np.linalg.pinv(L.T) @ K_s.T
            
            var = np.diag(K_ss) - np.sum(V**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        def expected_improvement(x, mu, var, best_f, xi=0.01):
            # x: (d,)
            # mu, var: scalars
            if var < 1e-10:
                return 0.0
            sigma = np.sqrt(var)
            z = (best_f - mu - xi) / sigma
            # Normal CDF and PDF
            # Using scipy.stats for precision if available, else approx
            try:
                from scipy.stats import norm
                cdf_z = norm.cdf(z)
                pdf_z = norm.pdf(z)
            except ImportError:
                # Approximation if scipy.stats not available (unlikely, but safe)
                cdf_z = 0.5 * (1 + np.tanh(z / np.sqrt(2))) # Rough approx, not great
                pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = (best_f - mu - xi) * cdf_z + sigma * pdf_z
            return ei

        def acquisition(x):
            # x: (d,)
            # Returns negative EI for minimization
            x = x.reshape(1, -1)
            mu, var = gp_predict(x, L, alpha, np.array(X), np.array(Y))
            ei = expected_improvement(x[0], mu[0], var[0], best_f)
            return -ei

        # Phase 1: Initialization
        n_init = 10
        for _ in range(n_init):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            f = wrapped_func(x)
            X.append(x)
            Y.append(f)

        if not X:
            return best_x, best_f

        X = np.array(X)
        Y = np.array(Y)

        # Phase 2: Bayesian Optimization Loop
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Fit GP
            L, alpha = fit_gp(X, Y)
            
            # Find next point by maximizing EI
            # Use multi-start local search for the acquisition function
            best_ei = -np.inf
            next_x = None
            
            n_acq_starts = 5
            for _ in range(n_acq_starts):
                # Start from best known points or random
                if rng.random() < 0.5 and len(X) > 0:
                    start_x = X[rng.integers(0, len(X))]
                else:
                    start_x = rng.uniform(-5, 5, self.dim)
                
                try:
                    res = minimize(
                        acquisition,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    if res.success or (hasattr(res, 'fun') and res.fun < -best_ei):
                        # Check if this is a better EI
                        ei_val = -res.fun
                        if ei_val > best_ei:
                            best_ei = ei_val
                            next_x = res.x
                except Exception:
                    pass
            
            if next_x is None:
                # Fallback to random
                next_x = rng.uniform(-5, 5, self.dim)
            
            # Evaluate next point
            f = wrapped_func(next_x)
            
            # Update data
            X = np.vstack([X, next_x.reshape(1, -1)])
            Y = np.append(Y, f)

        return best_x, best_f
