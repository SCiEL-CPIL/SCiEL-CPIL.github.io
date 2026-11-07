# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Surrogate-based optimizer using a Gaussian Process model
       with optimized hyperparameters and expected improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage for observations
        X = []
        Y = []

        def add_obs(x, y):
            X.append(np.asarray(x, dtype=float))
            Y.append(float(y))

        # Initial sampling: Sobol/Latin Hypercube style for diversity
        n_init = min(20, budget // 5)
        
        # Simple LHS initialization
        for i in range(n_init):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 2.0)
            elif i == 2:
                x0 = np.full(dim, -2.0)
            else:
                # Latin Hypercube sampling
                x0 = np.array([rng.uniform(0, 1) for _ in range(dim)])
                # Shuffle and scale to [-5, 5]
                # A simple permutation-based LHS
                perms = [np.arange(dim) for _ in range(dim)]
                for p in perms:
                    np.random.shuffle(p) # Use rng for reproducibility? rng.shuffle isn't direct.
                # Better LHS:
                x0 = np.zeros(dim)
                for d in range(dim):
                    x0[d] = (i + rng.random()) / n_init * 10.0 - 5.0
            fx = func(x0)
            add_obs(x0, fx)

        # Track best
        best_x = X[0].copy()
        best_f = Y[0]
        for i in range(1, len(X)):
            if Y[i] < best_f:
                best_f = Y[i]
                best_x = X[i].copy()

        # GP Hyperparameters - Initialize reasonably
        length_scale = 1.5
        variance = 1.0
        noise = 1e-5

        def compute_K(X1, X2, ls):
            """Compute Gaussian RBF kernel matrix."""
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0)
            return np.exp(-d2 / (2 * ls**2))

        def gp_predict(x_new, X_train, Y_train, ls, var, noise):
            """
            Predict mean and std at x_new using GP with RBF kernel.
            Returns (mu, sigma).
            """
            X_arr = np.array(X_train)
            Y_arr = np.array(Y_train)
            n = len(X_train)
            
            if n == 0:
                return 0.0, np.sqrt(var)

            # Kernel matrix K = K(X_train, X_train)
            K = compute_K(X_arr, X_arr, ls)
            K += noise * np.eye(n)
            
            # Kernel vector k = K(X_train, x_new)
            x_new_arr = x_new.reshape(1, -1)
            k = compute_K(X_arr, x_new_arr, ls).flatten()
            
            try:
                # Solve for alpha
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                
                # Predicted mean
                mu = k @ alpha
                
                # Predicted variance
                v = np.linalg.solve(L, k)
                sigma2 = var - k @ v
                sigma2 = max(sigma2, 1e-10)
                sigma = np.sqrt(sigma2)
            except np.linalg.LinAlgError:
                # Fallback: use simple weighted average
                d2 = np.sum((X_arr - x_new)**2, axis=1)
                K = np.exp(-d2 / (2 * ls**2))
                if np.sum(K) < 1e-12:
                    mu = np.mean(Y_arr)
                    sigma = np.std(Y_arr) + 0.5
                else:
                    mu = np.sum(K * Y_arr) / np.sum(K)
                    sigma = np.std(Y_arr) * (1 - np.max(K)) + 0.1
                    sigma = max(sigma, 0.01)
            
            return mu, sigma

        def negative_log_likelihood(params):
            """
            Negative log marginal likelihood for hyperparameter optimization.
            params: [log_length_scale, log_variance]
            """
            log_ls, log_var = params
            ls = np.exp(log_ls)
            var = np.exp(log_var)
            
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            n = len(X_arr)
            
            K = compute_K(X_arr, X_arr, ls)
            K += noise * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_arr))
                
                # Negative log likelihood
                nll = 0.5 * Y_arr @ alpha
                nll += np.sum(np.log(np.diag(L)))
                nll += 0.5 * n * np.log(2 * np.pi)
                return nll
            except np.linalg.LinAlgError:
                return 1e10

        def optimize_hyperparameters():
            """Optimize GP hyperparameters using L-BFGS-B on log-scale."""
            if len(X) < 5:
                return length_scale, variance
            
            # Initial guess in log space
            log_ls_init = np.log(length_scale)
            log_var_init = np.log(variance)
            
            # Bounds for log_ls and log_var
            # ls in [0.1, 5.0] -> log_ls in [-2.3, 1.6]
            # var in [0.01, 10.0] -> log_var in [-4.6, 2.3]
            bounds = [(-2.3, 1.6), (-4.6, 2.3)]
            
            res = minimize(
                negative_log_likelihood,
                [log_ls_init, log_var_init],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 20}
            )
            
            if res.success:
                length_scale = np.exp(res.x[0])
                variance = np.exp(res.x[1])
            
            return length_scale, variance

        def acquisition_ei(x):
            """
            Expected Improvement acquisition function.
            """
            mu, sigma = gp_predict(x, X, Y, length_scale, variance, noise)
            
            if sigma < 1e-10:
                return max(0, best_f - mu)
            
            z = (best_f - mu) / sigma
            # Standard normal CDF and PDF
            Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
            phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = (best_f - mu) * Phi + sigma * phi
            return ei

        # Optimization loop
        calls_made = len(X)
        
        for step in range(budget - calls_made):
            # Optimize hyperparameters every 5 steps
            if step % 5 == 0 and len(X) >= 5:
                length_scale, variance = optimize_hyperparameters()

            # Select next point by optimizing acquisition function
            # Use Differential Evolution for robust global search of EI
            # DE is slower but more robust than L-BFGS with random starts for non-convex EI
            # For dim=5, a small population is sufficient
            
            def neg_ei(z):
                return -acquisition_ei(z)

            try:
                res = differential_evolution(
                    neg_ei,
                    bounds=[(-5, 5)] * dim,
                    seed=int(rng.integers(0, 2**31)),
                    maxiter=10,
                    popsize=5,
                    tol=0.01,
                    init='sobol'
                )
                best_x_next = np.clip(res.x, -5, 5)
            except:
                # Fallback to random if DE fails
                best_x_next = rng.uniform(-5, 5, dim)

            # Evaluate the function
            f_new = func(best_x_next)
            add_obs(best_x_next, f_new)
            calls_made += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_next.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
