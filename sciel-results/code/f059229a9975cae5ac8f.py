import numpy as np
import time
from math import gamma, sqrt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global exploration with structured + random sampling
        # Büche-Rastrigin has many local minima; we need broad coverage
        # Use a mix of uniform random and quasi-random (Halton-like) points
        
        n_global = 50  # Use 50 calls for global phase
        
        # Add origin
        f0 = call_func(np.zeros(dim))
        if f0 is None:
            return best_x, best_f

        # Add axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                f = call_func(x)
                if f is None:
                    break
            if f is None:
                break
        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Halton sequence for quasi-random sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_global - 1 - 2 * dim + 1):
            if calls_made >= budget or not time_ok():
                break
            x = np.array([halton(i, b) for b in bases[:dim]])
            x = x * 8.0 - 4.0  # Map [0,1]^d to [-4,4]^d
            f = call_func(x)
            if f is None:
                break

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: CMA-ES with adaptive restarts
        # CMA-ES adapts covariance matrix to handle rotated landscapes efficiently
        # while keeping multi-start for global exploration
        
        remaining = budget - calls_made
        n_restarts = 4
        budget_per_restart = max(20, remaining // n_restarts)
        
        for restart in range(n_restarts):
            if calls_made >= budget or not time_ok():
                break
            
            # Initialize this restart
            if restart == 0:
                # Start from best found so far
                m = best_x.copy()
            else:
                # Mix: sometimes perturb best, sometimes sample fresh
                if rng.random() < 0.4:
                    m = best_x + rng.normal(0, 1.5, dim)
                else:
                    m = rng.uniform(-4.0, 4.0, dim)
                m = np.clip(m, -5.0, 5.0)
            
            f = call_func(m)
            if f is None:
                break
            
            local_best_f = f
            local_best_x = m.copy()
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            
            lambda_ = 12
            mu = lambda_ // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1 / np.sum(w**2)
            
            # Rates
            cs = (mueff + 2) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            cmu = min(1 - cs, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
            
            # Chi mean
            chi_mean = sqrt(2) * gamma((dim + 1) / 2) / gamma(dim / 2)
            
            # CMA-ES loop for this restart
            while calls_made < budget and time_ok():
                m_old = m.copy()
                
                # Generate samples
                xs = []
                ys = []
                fs = []
                
                for _ in range(lambda_):
                    if calls_made >= budget or not time_ok():
                        break
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    
                    z = rng.standard_normal(dim)
                    y = L @ z
                    x = m_old + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    
                    f_val = call_func(x)
                    if f_val is not None:
                        xs.append(x)
                        ys.append(y)
                        fs.append(f_val)
                
                if len(xs) < mu:
                    break
                
                # Sort by fitness
                indices = np.argsort(fs)
                xs_sorted = [xs[i] for i in indices[:mu]]
                ys_sorted = [ys[i] for i in indices[:mu]]
                fs_sorted = [fs[i] for i in indices[:mu]]
                
                # Update Mean
                m = np.zeros(dim)
                for i in range(mu):
                    m += w[i] * xs_sorted[i]
                
                # Update Cumulation Vector
                y_1 = ys_sorted[0]
                pc = (1 - cs) * pc + cs * np.sqrt(mueff) * y_1
                
                # Update Sigma
                sigma *= np.exp((cs / ds) * (np.linalg.norm(pc) / chi_mean - 1))
                
                # Update Covariance
                sum_ww_yy = np.zeros((dim, dim))
                for i in range(mu):
                    sum_ww_yy += w[i] * np.outer(ys_sorted[i], ys_sorted[i])
                    
                C = (1 - cs - cmu) * C + cs * np.outer(pc, pc) + cmu * sum_ww_yy
                
                # Ensure C is positive semi-definite (regularization)
                C = (C + C.T) / 2
                
                # Track local best
                if fs_sorted[0] < local_best_f:
                    local_best_f = fs_sorted[0]
                    local_best_x = xs_sorted[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                
                # Stagnation check: if no improvement in last 30 evaluations
                if local_best_f >= best_f and (calls_made - (budget - budget_per_restart)) > 30:
                    # Check if we've made progress
                    pass
                
                # Minimum sigma guard
                if sigma < 0.005:
                    break

        return best_x, best_f
