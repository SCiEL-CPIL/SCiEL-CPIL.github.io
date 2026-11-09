import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 310.0  # Safety margin (330s kill, leave 20s)

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        
        n = self.dim
        
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return float(func(np.clip(x, -5.0, 5.0)))
        
        best_x = np.zeros(n)
        best_fx = float('inf')
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 1: Coarse global search with Nelder-Mead multi-start
        # Bent cigar is anisotropic; Nelder-Mead adapts simplex shape to find the long axis.
        # We start from several diverse points to cover the shifted domain.
        
        # Generate starting points
        start_points = [np.zeros(n)]
        
        # Structured points along axes
        for i in range(n):
            p_pos = np.zeros(n)
            p_pos[i] = 3.0
            start_points.append(p_pos)
            
            p_neg = np.zeros(n)
            p_neg[i] = -3.0
            start_points.append(p_neg)
        
        # Random points
        rng = np.random.RandomState(42)
        for _ in range(8):
            start_points.append(rng.uniform(-4.5, 4.5, size=n))
        
        # Shuffle to mix structured and random
        indices = list(range(len(start_points)))
        rng.shuffle(indices)
        start_points = [start_points[i] for i in indices]
        
        from scipy.optimize import minimize
        
        for i, x0 in enumerate(start_points):
            if self.calls >= self.budget - 20:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            
            try:
                # Nelder-Mead with adaptive tolerance
                # maxiter: each iteration does ~n function calls
                # We allow a decent number of iterations for the coarse phase
                max_iter = min(100, max(10, (self.budget - self.calls - 10) // (n + 1)))
                
                res = minimize(eval_func, x0, method='Nelder-Mead',
                              options={
                                  'maxiter': max_iter,
                                  'xatol': 1e-8,
                                  'fatol': 1e-10,
                                  'adaptive': True
                              })
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: Fine-tune with CMA-ES from the best point found
        # CMA-ES is robust for continuous optimization and handles anisotropy well.
        remaining = self.budget - self.calls
        if remaining > 20 and best_x is not None and time.time() - self.start_time < self.time_limit:
            try:
                # Use CMA-ES with a smaller initial sigma for fine-tuning
                # Population size: 4 + floor(3 * log(n))
                lam = 4 + int(3 * np.log(n))
                
                # Initial sigma: small, since we're refining
                sigma0 = 0.1
                
                # Manual CMA-ES implementation to have full control over budget
                mean = best_x.copy()
                sigma = sigma0
                cov = np.eye(n)
                
                # CMA-ES parameters
                c_sigma = (lam + 2) / (n + lam + 4)
                c_cov = (2 + 2 * np.sqrt((lam - 2) / (n + 2))) / (n + 3)
                mu = lam // 2
                # Weights
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1.0 / np.sum(weights ** 2)
                c_c = (mu_eff - 2 + 1 / mu_eff) / ((n + 2) + mu_eff - 2)
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
                
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                
                # Initial population
                rng_cma = np.random.RandomState(123)
                
                max_cma_iter = min(50, max(5, remaining // lam))
                
                for it in range(max_cma_iter):
                    if self.calls >= self.budget:
                        break
                    if time.time() - self.start_time > self.time_limit:
                        break
                    
                    # Generate population
                    W = np.linalg.cholesky(cov)
                    ys = []
                    xs = []
                    fs = []
                    
                    for j in range(lam):
                        if self.calls >= self.budget:
                            break
                        if time.time() - self.start_time > self.time_limit:
                            break
                        y = rng_cma.standard_normal(n)
                        x = mean + sigma * (W @ y)
                        x_clipped = np.clip(x, -5.0, 5.0)
                        f = eval_func(x_clipped)
                        ys.append(y)
                        xs.append(x_clipped)
                        fs.append(f)
                    
                    if len(fs) < lam:
                        break
                    
                    # Sort by fitness
                    idx = np.argsort(fs)
                    xs_sorted = [xs[i] for i in idx]
                    ys_sorted = [ys[i] for i in idx]
                    fs_sorted = [fs[i] for i in idx]
                    
                    # Update best
                    if fs_sorted[0] < best_fx:
                        best_fx = fs_sorted[0]
                        best_x = xs_sorted[0].copy()
                    
                    # Compute new mean
                    y_w = sum(weights[j] * ys_sorted[j] for j in range(mu))
                    x_w = sum(weights[j] * xs_sorted[j] for j in range(mu))
                    
                    # Update evolution paths
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (W @ y_w)
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (sigma * W @ y_w)
                    
                    # Update covariance
                    delta = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)) else 0
                    cov = (1 + c_cov * (mu_eff - 1) * delta - c_cov) * cov \
                          + c_cov * np.outer(y_w, y_w) * (sigma ** 2)
                    
                    # Update sigma
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt((n + 1) / (n - 1)) - 1))
                    
                    # Update mean
                    mean = x_w
                    
                    # Ensure cov is symmetric and positive semi-definite
                    cov = (cov + cov.T) / 2
                    try:
                        W = np.linalg.cholesky(cov)
                    except:
                        # Regularize
                        cov += 1e-6 * np.eye(n)
                        W = np.linalg.cholesky(cov)
                        
            except StopIteration:
                break
            except Exception:
                pass
        
        # Phase 3: Final local refinement with Nelder-Mead
        remaining = self.budget - self.calls
        if remaining > 5 and best_x is not None and time.time() - self.start_time < self.time_limit:
            try:
                max_iter = min(50, max(5, remaining // (n + 1)))
                res = minimize(eval_func, best_x, method='Nelder-Mead',
                              options={
                                  'maxiter': max_iter,
                                  'xatol': 1e-10,
                                  'fatol': 1e-12,
                                  'adaptive': True
                              })
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        if best_x is None:
            best_x = np.zeros(n)
        
        return best_x, best_fx
