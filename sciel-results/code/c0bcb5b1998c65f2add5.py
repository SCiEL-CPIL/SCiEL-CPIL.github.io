import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-inspired adaptive population-based search
        # This adapts the covariance matrix to learn the shape of the fitness landscape
        
        # Initial parameters
        pop_size = 20  # population size
        sigma = 2.0    # initial step size
        mean = np.zeros(d)
        
        # Covariance matrix (start with identity)
        C = np.eye(d)
        # Eigen-decomposition for efficient sampling
        eigval, eigvec = np.linalg.eigh(C)
        eigval = np.sqrt(np.maximum(eigval, 1e-12))
        
        # Adaptation rates
        cc = (4 + d / 3.0) / (d + 4.0 + 2.0 / (d + 1.0))  # mean adaptation
        cs = (4 + d / 2.0) / (d + 12.0 + 2.0 / (d + 1.0))  # sigma adaptation
        c1 = 2 / ((d + 1.3)**2 + d)  # covariance adaptation (rank-1)
        cmu = min(1 - c1, 2 * (mu0 - 0.5 + 1.0 / d) / ((d + 2.0)**2 + d))  # covariance adaptation (rank-mu)
        
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu0 = weights[0] ** 2 / (weights ** 2).sum()
        
        # Phase 1: Initial broad exploration
        n_initial = 15
        for i in range(n_initial):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            call_func(x)
        
        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)
        
        # Phase 2: Adaptive population-based search
        while calls_made < self.budget and time.time() < deadline:
            # Generate population from adapted distribution
            population = []
            for i in range(pop_size):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal with adapted covariance
                z = rng.standard_normal(d)
                # Transform using eigendecomposition
                x_sample = mean + sigma * (eigvec * eigval) @ z
                x_sample = np.clip(x_sample, lb, ub)
                
                f = call_func(x_sample)
                population.append((f, x_sample))
            
            if len(population) < mu:
                break
            
            # Sort by fitness
            population.sort(key=lambda item: item[0])
            
            # Update mean (weighted combination of best mu individuals)
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]
            
            # Update step size (sigma)
            # Simple heuristic: adapt based on improvement
            if best_f < np.inf:
                # Scale sigma based on how well we're doing
                # If we're making progress, keep sigma; if stuck, reduce
                pass
            
            # Update covariance matrix
            # Rank-1 update
            diff = new_mean - mean
            C += (c1 * (1 + (1 - cs) * cc) * d + c1 * cc * (1 - cc)) * np.outer(diff, diff)
            
            # Rank-mu update
            for i in range(mu):
                diff_i = population[i][1] - mean
                C += cmu * weights[i] * np.outer(diff_i, diff_i)
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2.0
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.sqrt(np.maximum(eigval, 1e-12))
            
            # Update mean
            mean = new_mean
            
            # Adapt sigma: if we're stuck, reduce; if improving, keep
            # Simple strategy: slowly decay sigma to focus search
            sigma *= 0.98
            
            # If sigma becomes too small, do a restart
            if sigma < 0.01:
                # Restart with larger sigma around current best
                sigma = 1.0
                mean = best_x.copy()
                C = np.eye(d)
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.sqrt(np.maximum(eigval, 1e-12))
            
            # Check for time
            if time.time() > deadline - 5.0:
                break
        
        # Phase 3: Final local refinement if budget remains
        remaining = self.budget - calls_made
        if remaining > 10 and best_x is not None and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    call_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxfev': min(remaining, 50), 'ftol': 1e-12}
                )
            except Exception:
                pass
        
        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
