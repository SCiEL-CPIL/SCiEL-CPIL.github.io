"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for 5-dimensional BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * np.log(n))  # population size
        cc = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        cs = (n + 2 + cs_ / cc) / (n + 5 + cs_ / cc) if (cs_ := 1.0) else 0
        # Standard CMA-ES params
        cs = (n + 2 + 1.0 / cc) / (n + 5 + 1.0 / cc)
        dm = 1 + 2 * max(0, np.sqrt((cs + 1) / (cs + 1)) - 1) + cs
        cmu = min(1 - 1 / dm, 2 * ((lambda_ + 2) / (n + 2) - 1) / (dm * (dm + n + 2)))
        sigma0 = 1.0
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial center: try a few random starts, pick the best
        best_x = None
        best_f = np.inf
        for _ in range(5):
            x0 = rng.uniform(-5, 5, n)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        x = best_x
        sigma = sigma0
        B = np.eye(n)
        D = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls = 5
        deadline = time.time() + 290  # Safety margin
        
        while calls < self.budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D @ z))
                y = np.clip(y, -5, 5)
                pop.append(y)
            
            # Evaluate population
            fpop = [func(y) for y in pop]
            calls += lambda_
            
            # Sort
            order = np.argsort(fpop)
            pop = [pop[i] for i in order]
            fpop = [fpop[i] for i in order]
            
            # Update best
            if fpop[0] < best_f:
                best_f = fpop[0]
                best_x = pop[0].copy()
            
            # Update mean
            weights = np.full(lambda_, 1 / (2 * lambda_))
            weights[:int(np.ceil(lambda_ / 2))] += np.log(lambda_ / 2) - np.log(lambda_)
            weights /= np.sum(weights)
            
            x_new = np.sum([w * y for w, y in zip(weights, pop)], axis=0)
            
            # Update evolution paths
            d = (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * dm) * d
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - 4))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            pc = (1 - cc) * pc + hsig * cc * d
            
            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C_update = (1 - (cmu + delta_hsig)) * np.eye(n)
            for w, y in zip(weights, pop):
                C_update += w * np.outer((y - x) / sigma, (y - x) / sigma)
            C_update += delta_hsig * np.outer(pc, pc)
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_update)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            x = x_new
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # If stuck, try restart
            if calls % 50 == 0 and np.std(fpop[:int(lambda_/2)]) < 1e-6:
                # Try a new random start
                x0 = rng.uniform(-5, 5, n)
                f0 = func(x0)
                calls += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                x = x0
                sigma = 1.0
                pc = np.zeros(n)
                ps = np.zeros(n)
                B = np.eye(n)
                D = np.eye(n)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
