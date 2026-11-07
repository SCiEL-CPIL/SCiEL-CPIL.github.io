"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate for low-dimensional (d=5)
    Schaffers F7 optimization with budget 500.
    
    Strategy:
    1. Initial random sampling to build a diverse set of observations.
    2. GP surrogate model with Expected Improvement acquisition function.
    3. Iteratively select next point via EI, evaluate, update GP.
    4. Hybrid: periodically use Nelder-Mead on the GP-predicted minimum for exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X, X_test):
        """
        Gaussian Process prediction.
        X: observed points (n x d), y: observed values
        X_test: points to predict (m x d)
        Returns: mean (m,), std (m,)
        """
        # We'll use a simple squared exponential kernel with fixed hyperparameters
        # to avoid the overhead of hyperparameter optimization.
        alpha = 1.0  # length scale
        sigma_f = 1.0  # signal variance
        sigma_n = 1e-6  # noise variance
        
        # Compute kernel matrix K(X, X)
        def kernel(A, B):
            # A: n x d, B: m x d
            # Compute pairwise squared distances
            A2 = np.sum(A**2, axis=1).reshape(-1, 1)
            B2 = np.sum(B**2, axis=1).reshape(1, -1)
            D2 = A2 + B2 - 2 * A @ B.T
            D2 = np.maximum(D2, 0)  # numerical stability
            return sigma_f**2 * np.exp(-0.5 * D2 / alpha**2)
        
        n = X.shape[0]
        K = kernel(X, X) + sigma_n * np.eye(n)
        
        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        
        # Solve L @ alpha = y
        y = self._y.copy()
        alpha_vec = np.linalg.solve(L, y)
        alpha_vec = np.linalg.solve(L.T, alpha_vec)
        
        # Predict at X_test
        K_s = kernel(X_test, X)
        mu = K_s @ alpha_vec
        v = np.linalg.solve(L, K_s.T)
        var = sigma_f**2 - np.sum(v**2, axis=1)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)
        
        return mu, std

    def __call__(self, func):
        calls = 0
        max_calls = self.budget

        def safe_call(x):
            nonlocal calls
            if calls >= max_calls:
                return np.inf
            calls += 1
            return func(x)

        rng = np.random.default_rng()
        
        # Phase 1: Initial random sampling
        # Use ~30% of budget for initial exploration
        n_init = max(20, self.budget // 4)
        
        self._X = []
        self._y = []
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            self._X.append(x.copy())
            self._y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        self._X = np.array(self._X)
        self._y = np.array(self._y)
        
        if calls >= max_calls:
            return best_x, best_f

        # Phase 2: Bayesian Optimization loop
        # Use remaining budget for BO iterations
        # Each iteration: 1 function call
        # We'll run BO until budget is exhausted, but periodically
        # switch to Nelder-Mead exploitation for the last portion.
        
        n_bo = max_calls - calls
        n_nm_reserve = max(20, n_bo // 5)  # Reserve some budget for final NM polish
        n_bo_actual = n_bo - n_nm_reserve
        
        for i in range(n_bo_actual):
            if calls >= max_calls:
                break
            
            # Get GP predictions on a candidate set
            # Use a mix of random candidates and local candidates around best_x
            n_candidates = 200
            candidates = []
            
            # 70% random in the box
            n_rand = int(0.7 * n_candidates)
            candidates.append(rng.uniform(-5, 5, (n_rand, self.dim)))
            
            # 30% local around best_x
            n_local = n_candidates - n_rand
            local_sigma = 1.0
            local_pts = best_x + local_sigma * rng.standard_normal((n_local, self.dim))
            local_pts = np.clip(local_pts, -5, 5)
            candidates.append(local_pts)
            
            candidates = np.vstack(candidates)
            
            # GP prediction
            mu, std = self._gp_predict(self._X, candidates)
            
            # Expected Improvement
            best_current = np.min(self._y)
            Z = (best_current - mu) / std
            # Handle std=0 cases
            valid = std > 1e-10
            EI = np.zeros_like(mu)
            EI[valid] = std[valid] * (Z[valid] * norm.cdf(Z[valid]) + norm.pdf(Z[valid]))
            EI[~valid] = np.maximum(best_current - mu[~valid], 0)
            
            # Select point with max EI
            idx = np.argmax(EI)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = safe_call(x_next)
            
            # Update observations
            self._X = np.vstack([self._X, x_next.reshape(1, -1)])
            self._y = np.append(self._y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        if calls >= max_calls:
            return best_x, best_f

        # Phase 3: Final Nelder-Mead polish
        remaining = max_calls - calls
        if remaining > 0:
            try:
                res = minimize(
                    safe_call,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
