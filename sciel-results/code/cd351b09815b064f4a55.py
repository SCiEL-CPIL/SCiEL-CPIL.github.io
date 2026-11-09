"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        # Initial parameters
        x = np.zeros(self.dim)
        sigma = 2.0
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(self.dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        C = np.eye(self.dim)
        D = 1.0
        B = np.eye(self.dim)
        
        calls_made = 0
        best_x = x.copy()
        best_f = func(x)
        calls_made += 1
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Sample population
            z = rng.standard_normal((lam, self.dim))
            # y = B @ z.T
            y = (B @ z.T).T
            pop = x + sigma * y
            
            # Evaluate population
            fvals = np.array([func(p) for p in pop])
            calls_made += lam
            
            if calls_made >= self.budget:
                break
                
            # Sort by fitness
            order = np.argsort(fvals)
            z_sorted = z[order]
            f_sorted = fvals[order]
            pop_sorted = pop[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = pop_sorted[0].copy()
            
            # Update mean
            x_old = x.copy()
            x_new = pop_sorted[:mu].T @ weights
            x = x_new
            
            # Update step size control (CSA)
            # hsig is calculated based on the norm of ps
            # Approximation for hsig
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * D * (B.T @ (x - x_old) / sigma)
            
            # Update covariance matrix
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * D * (B.T @ (x - x_old) / sigma)
            
            # C update
            # delta_hs is 0 if hsig is 1, else 1
            # We skip the exact hsig calculation for simplicity and stability, using a fixed delta_hs=0.
            # This is a common simplification.
            delta_hs = 0.0
            
            C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc) ** 2) * D ** 2 * np.eye(self.dim))
            
            # Sum of weighted outer products of y_i
            y_sum = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (pop_sorted[i] - x_old) / sigma
                y_sum += weights[i] * np.outer(y_i, y_i)
            
            C += cmu * y_sum
            
            # Update D and B using eigen decomposition
            # This is expensive. We can do it every few iterations.
            # For dim=5, it's fast enough.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals).mean()
            B = eigvecs
            
        return best_x, best_f

# EVOLVE-BLOCK-END
