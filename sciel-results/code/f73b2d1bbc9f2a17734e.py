"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # CMA-ES parameters
        self.pop_size = 4 + int(3 * math.log(self.dim))  # lambda
        self.step_size = 2.0
        self.cc = (4 + self.dim / self.pop_size) / (self.dim + 4 + 2.0 / self.pop_size)
        self.cs = (self.pop_size / 2 + 1) / (self.dim + self.pop_size / 2 + 3)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.pop_size)
        self.cmua = min(1 - self.c1, 2 / ((self.dim + 2) ** 2 + self.pop_size))
        self.damps = 1 + 2 * max(0, math.sqrt((self.pop_size - 1) / (self.dim + 1)) - 1) + self.cs
        self.maxiter = self.budget // self.pop_size

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial population
        C = np.eye(self.dim)
        sigma = self.step_size
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = math.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        best_x = x.copy()
        best_f = fx
        call_count = 1
        
        # Pre-generate some random numbers if possible? No, CMA-ES is adaptive.
        # We will run the CMA-ES loop.
        
        for _ in range(self.maxiter):
            if time.time() > deadline:
                break
            
            # Generate population
            D = np.linalg.cholesky(C)
            B = D.T
            # Inverse of B for step size control? No, standard CMA-ES update.
            
            # Sample
            z = rng.standard_normal((self.pop_size, self.dim))
            pop = x + sigma * (z @ B)
            
            # Evaluate
            fvals = []
            for i in range(self.pop_size):
                if time.time() > deadline:
                    # Fill remaining with current best to avoid errors if we break mid-eval?
                    # Better to just break and return what we have.
                    break
                xi = np.clip(pop[i], -5, 5)
                fi = func(xi)
                fvals.append((fi, i))
                call_count += 1
            
            if not fvals:
                break
                
            # Sort
            fvals.sort(key=lambda x: x[0])
            
            # Update best
            if fvals[0][0] < best_f:
                best_f = fvals[0][0]
                best_x = np.clip(pop[fvals[0][1]], -5, 5)
            
            # Select top half
            mu = self.pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.array([i + 1 for i in range(mu)]))
            w /= w.sum()
            mueff = 1 / (w @ w)
            
            # Update mean
            y_w = np.zeros(self.dim)
            for i in range(mu):
                idx = fvals[i][1]
                y_w += w[i] * (pop[idx] - x)
            
            x_new = x + y_w
            
            # Update covariance
            y_inv = y_w / sigma
            pc = (1 - self.cc) * pc + self.cc * y_inv
            ps = (1 - self.cs) * ps + math.sqrt(self.cs * (2 - self.cs) * mueff) * pc
            
            hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (call_count / self.pop_size)))) / chi_n
            hsig = min(1.0, hsig)
            
            C = (1 - self.c1 - self.cmua) * C
            C += self.c1 * (np.outer(pc, pc) + (hsig - 1 + self.cc * (1 - self.cc)) * C)
            C += self.cmua * (np.outer(y_inv, y_inv) - C)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update step size
            sigma *= math.exp(self.cs / self.damps * (np.linalg.norm(ps) / chi_n - 1))
            
            x = x_new
            
            # Check convergence
            if np.all(np.abs(x - x_new) < 1e-8):
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
