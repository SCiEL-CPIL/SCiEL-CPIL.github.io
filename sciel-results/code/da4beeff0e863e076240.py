# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin

        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            if time.perf_counter() > deadline:
                return float('inf')
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                return float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Radial Sweep Initialization
        # Sample points along axes and random directions to find the needle
        directions = []
        for i in range(dim):
            for sign in [1, -1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        
        for _ in range(5):
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                d /= norm
                directions.append(d)
                
        radii = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        for d in directions:
            for r in radii:
                if calls_made >= budget or time.perf_counter() > deadline:
                    break
                x = np.clip(r * d, -5, 5)
                f = evaluate(x)
                if f == float('inf'):
                    break

        if calls_made >= budget or time.perf_counter() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES
        # Initialize CMA-ES state
        m = best_x.copy()
        sigma = 1.0 # Initial step size, maybe adjust based on best_f?
        
        # CMA-ES parameters for dim=5
        lam = 4  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        
        cc = (4 + dim / lam) / ((dim + 4) * (lam + 1.7))
        cs = (lam + 2 + cc) / ((dim + lam + 5) * (lam + 1.7 - cc))
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + lam))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky and inverse sqrt
        Cholesky = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        
        i = 0
        while calls_made < budget and time.perf_counter() < deadline:
            # Generate population
            ys = []
            for j in range(lam):
                if calls_made >= budget or time.perf_counter() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = m + sigma * (Cholesky @ z)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy == float('inf'):
                    break
                ys.append((fy, y))
            
            if not ys:
                break
            
            # Sort by fitness
            ys.sort(key=lambda t: t[0])
            
            # Update mean
            y_mean = np.zeros(dim)
            for k in range(mu):
                y_mean += weights[k] * ys[k][1]
            
            # Update step size evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ (y_mean - m) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (i + 1)) / np.sqrt(dim) < 1.4 + 2 / (dim + 1))
            
            # Update step size
            sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_mean - m) / sigma
            
            # Update mean
            m = y_mean
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - (1 - delta_hsig) * C)
            for k in range(mu):
                yk = ys[k][1] - m
                C += cmu * weights[k] * np.outer(yk, yk)
            
            C = (C + C.T) / 2
            
            # Update Cholesky and inverse sqrt periodically
            if i % 5 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, 1e10)
                    sqrt_eigvals = np.sqrt(eigvals)
                    Cholesky = eigvecs @ (sqrt_eigvals[:, None] * eigvecs.T)
                    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                    Cinv_sqrt = eigvecs @ (inv_sqrt_eigvals[:, None] * eigvecs.T)
                except np.linalg.LinAlgError:
                    # Reset if numerical issues
                    C = np.eye(dim)
                    Cholesky = np.eye(dim)
                    Cinv_sqrt = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
            
            i += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
