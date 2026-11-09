import numpy as np
import time
from scipy.stats import norm
import scipy.optimize as optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 330s kill limit. Keep a safety margin.
        deadline = start_time + 320.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # 1. Global Exploration: Sobol/Halton-like initial sampling
        # We need enough points to initialize the GP and find a decent basin.
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Generate quasi-random points in [0, 1]^dim
        # Simple Halton for 5D
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n_init):
                X_init[i, d] = halton(i + 1, base)
        
        # Map to [-5, 5]
        X_init = X_init * 10.0 - 5.0
        
        # Evaluate initial points
        X_obs = []
        y_obs = []
        best_x = None
        best_f = np.inf
        
        for i in range(n_init):
            if not time_ok():
                break
            x = X_init[i]
            fx = func(x)
            X_obs.append(x)
            y_obs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        calls_made = len(X_obs)
        
        if calls_made == 0:
            return None, np.inf

        # 2. Surrogate-Assisted Bayesian Optimization Loop
        # We use a Gaussian Process with Matern 5/2 kernel.
        # Since scipy doesn't have a built-in efficient GP optimizer for acquisition
        # that is super fast for 5D with 500 budget, we need to be careful.
        
        # However, 500 calls is small. Fitting a GP on ~500 points in 5D is fast.
        # The bottleneck is optimizing the acquisition function.
        # We will use a simple grid + local optimization for the acquisition step.
        
        # Parameters
        # We will refit the GP every 5-10 steps to keep it fresh and computationally cheap?
        # Or just every step? With N=500, N^3 is 125M ops, which is fine in numpy/scipy.
        # Let's try refitting every step for accuracy, but monitor time.
        
        # Kernel: Matern 5/2
        # K(x, x') = sigma_f^2 * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
        # where d = ||x - x'|| / l
        
        class Matern52GP:
            def __init__(self, X, y, l=1.0, sigma_f=1.0, sigma_n=1e-5):
                self.X = X
                self.y = y
                self.l = l
                self.sigma_f = sigma_f
                self.sigma_n = sigma_n
                self.n = len(y)
                self._fit()
            
            def _kernel(self, X1, X2):
                # X1: (n1, dim), X2: (n2, dim)
                # Compute squared distances
                X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
                X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
                d_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
                np.maximum(d_sq, 0, out=d_sq)
                d = np.sqrt(d_sq) / self.l
                return self.sigma_f**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)
            
            def _fit(self):
                K = self._kernel(self.X, self.X)
                K += self.sigma_n**2 * np.eye(self.n)
                # Add jitter for numerical stability
                K += 1e-6 * np.eye(self.n)
                try:
                    self.L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if Cholesky fails
                    K_inv = np.linalg.pinv(K)
                    self.alpha = K_inv @ self.y
                    self.L = None # Mark as using pseudo-inverse
            
            def predict(self, X_new):
                # Returns mean and std
                if self.L is None:
                    # Pseudo-inverse path (less stable, but robust)
                    K_s = self._kernel(self.X, X_new)
                    mean = K_s.T @ self.alpha
                    # Variance calculation is tricky with pinv, approximate
                    var = self.sigma_f**2 - np.sum(K_s * K_s.T @ self.alpha, axis=1)
                    var = np.maximum(var, 1e-12)
                    return mean, np.sqrt(var)
                
                K_s = self._kernel(self.X, X_new)
                v = np.linalg.solve(self.L, K_s.T)
                mean = K_s @ self.alpha
                var = self.sigma_f**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
                return mean, np.sqrt(var)
        
        # Acquisition: Expected Improvement (EI)
        # EI(x) = (mu - f_best - xi) * Phi((mu - f_best - xi)/sigma) + sigma * phi((mu - f_best - xi)/sigma)
        # For minimization
        
        def expected_improvement(X, mu, sigma, f_best, xi=0.01):
            # X: (n, dim)
            # mu, sigma: (n,)
            z = (mu - f_best - xi) / sigma
            ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei
        
        # Optimization of acquisition function
        # We will use a multi-start local optimization (Nelder-Mead) on the acquisition function.
        
        gp = Matern52GP(X_obs, y_obs, l=1.0, sigma_f=1.0, sigma_n=1e-5)
        
        # Adaptive length scale? 
        # For now, fix l=1.0. If performance is poor, we might need to tune it,
        # but the task asks for a structural change, not tuning.
        
        while calls_made < budget and time_ok():
            # 1. Predict on a grid or random points to find good candidates
            # Since dim=5, a dense grid is expensive. 
            # Strategy: Sample 100 random points, evaluate EI, pick top 5, then NM refine.
            
            # Check time before heavy computation
            if not time_ok():
                break
                
            # Sample random points in [-5, 5]^5
            n_candidates = 100
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            mu, sigma = gp.predict(X_cand)
            ei = expected_improvement(X_cand, mu, sigma, best_f, xi=0.01)
            
            # Select top K candidates for local refinement
            K_refine = 5
            top_idx = np.argsort(ei)[-K_refine:]
            X_start = X_cand[top_idx]
            EI_start = ei[top_idx]
            
            best_x_new = None
            best_ei_new = -np.inf
            
            # Local optimization of EI using Nelder-Mead
            # We need to define the objective for NM: -EI(x)
            
            def neg_ei(x):
                x = np.array(x)
                if np.any(x < -5.01) or np.any(x > 5.01):
                    return np.inf
                x = np.clip(x, -5, 5)
                m, s = gp.predict(x.reshape(1, -1))
                e = expected_improvement(x.reshape(1, -1), m, s, best_f, xi=0.01)[0]
                return -e
            
            # Run NM on top candidates
            for i in range(K_refine):
                if not time_ok():
                    break
                try:
                    res = optimize.minimize(neg_ei, X_start[i], method='Nelder-Mead', 
                                            options={'maxiter': 50, 'xatol': 0.05, 'fatol': 1e-4})
                    if res.fun < best_ei_new:
                        best_ei_new = res.fun
                        best_x_new = res.x
                except Exception:
                    continue
            
            if best_x_new is None:
                # Fallback to best candidate from grid
                best_idx = np.argmax(ei)
                best_x_new = X_cand[best_idx]
            
            # Clip to box
            best_x_new = np.clip(best_x_new, -5, 5)
            
            # Evaluate the new point
            f_new = func(best_x_new)
            calls_made += 1
            
            X_obs = np.vstack([X_obs, best_x_new.reshape(1, -1)])
            y_obs = np.append(y_obs, f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_new.copy()
            
            # Refit GP
            # To save time, we can refit every step, but if it gets slow, we can skip.
            # Given N=500 max, it should be fine.
            try:
                # Simple adaptive noise: if points are close, reduce noise?
                # Keep sigma_n fixed for now.
                gp = Matern52GP(X_obs, y_obs, l=1.0, sigma_f=1.0, sigma_n=1e-5)
            except Exception:
                # If GP fitting fails, fall back to random search for remaining budget
                while calls_made < budget and time_ok():
                    x_rand = rng.uniform(-5, 5, size=dim)
                    f_rand = func(x_rand)
                    calls_made += 1
                    if f_rand < best_f:
                        best_f = f_rand
                        best_x = x_rand.copy()
                break
                
            # Check stagnation?
            # If we haven't improved in 20 steps, maybe restart GP or change kernel?
            # For now, rely on EI's exploration capability.
            
        return best_x, best_f
