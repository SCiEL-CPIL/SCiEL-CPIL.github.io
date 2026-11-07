import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import cho_solve
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # State
        X = np.array([]).reshape(0, self.dim)
        Y = np.array([])
        call_count = 0
        
        # Initial Design: Latin Hypercube Sampling
        n_init = min(15, self.budget)
        rng = np.random.default_rng(42)
        X_init = rng.uniform(-5, 5, (n_init, self.dim))
        
        for x in X_init:
            if call_count >= self.budget: break
            f = func(x)
            X = np.vstack([X, x])
            Y = np.append(Y, f)
            call_count += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x
            if time.time() > deadline:
                return self.best_x, self.best_f

        # GP Parameters
        def matern52_kernel(X1, X2, lengths, sigma2):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dists = np.sum(diff**2 / lengths**2, axis=-1)
            sqrt_sq_dists = np.sqrt(sq_dists)
            K = sigma2 * (1 + sqrt_sq_dists * np.sqrt(5) + (5/3) * sq_dists) * np.exp(-sqrt_sq_dists * np.sqrt(5))
            return K

        def log_marginal_likelihood(params, X_data, Y_data):
            if len(params) != self.dim + 2:
                return 0.0
            lengths = np.exp(params[:self.dim])
            sigma2 = np.exp(params[self.dim])
            noise2 = np.exp(params[self.dim + 1])
            
            K = matern52_kernel(X_data, X_data, lengths, sigma2)
            K += noise2 * np.eye(len(X_data))
            K += 1e-8 * np.eye(len(X_data))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return -np.inf
            
            alpha = cho_solve((L, True), Y_data)
            
            n = len(Y_data)
            lml = -0.5 * Y_data @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return lml

        # Main Loop
        while call_count < self.budget:
            if time.time() > deadline:
                break
                
            # 1. Optimize Hyperparameters
            init_params = np.zeros(self.dim + 2)
            init_params[:self.dim] = 0.0
            init_params[self.dim] = 0.0
            init_params[self.dim + 1] = -2.0
            
            # Limit iterations for speed
            res = minimize(
                log_marginal_likelihood, 
                init_params, 
                args=(X, Y), 
                method='L-BFGS-B', 
                options={'maxiter': 10, 'ftol': 1e-4}
            )
            
            if not res.success:
                lengths = np.exp(init_params[:self.dim])
                sigma2 = np.exp(init_params[self.dim])
                noise2 = np.exp(init_params[self.dim + 1])
            else:
                lengths = np.exp(res.x[:self.dim])
                sigma2 = np.exp(res.x[self.dim])
                noise2 = np.exp(res.x[self.dim + 1])
            
            # Precompute GP internals
            K = matern52_kernel(X, X, lengths, sigma2)
            K += noise2 * np.eye(len(X))
            K += 1e-8 * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = cho_solve((L, True), Y)
            except np.linalg.LinAlgError:
                # Fallback to random if GP fails
                x_next = np.random.uniform(-5, 5, self.dim)
                f_new = func(x_next)
                X = np.vstack([X, x_next])
                Y = np.append(Y, f_new)
                call_count += 1
                if f_new < self.best_f:
                    self.best_f = f_new
                    self.best_x = x_next
                continue

            # 2. Acquisition: Vectorized EI on random candidates
            f_min = np.min(Y)
            n_cand = 500
            X_cand = np.random.uniform(-5, 5, (n_cand, self.dim))
            
            K_cand = matern52_kernel(X_cand, X, lengths, sigma2) # (N_cand, N_data)
            
            # Mean
            mu_cand = K_cand @ alpha # (N_cand,)
            
            # Variance
            # v = L^{-T} K_cand.T
            # K_cand is (M, N). We need to solve L v^T = K_cand.
            # v^T is (N, M).
            # np.linalg.solve(L, K_cand.T) gives (N, M)
            v_cand = np.linalg.solve(L, K_cand.T) # (N_data, N_cand)
            sigma2_cand = sigma2 - np.sum(K_cand * v_cand.T, axis=1) # (N_cand,)
            sigma2_cand = np.maximum(sigma2_cand, 1e-8)
            sigma_cand = np.sqrt(sigma2_cand)
            
            # EI
            Z = (f_min - mu_cand) / sigma_cand
            ei_cand = (f_min - mu_cand) * norm.cdf(Z) + sigma_cand * norm.pdf(Z)
            
            # Find best candidate
            idx_best = np.argmax(ei_cand)
            x_next = X_cand[idx_best]
            
            # 3. Evaluate
            f_new = func(x_next)
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_new)
            call_count += 1
            
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_next
                
        return self.best_x, self.best_f
