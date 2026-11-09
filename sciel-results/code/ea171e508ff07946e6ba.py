"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Phase 1: Random restarts to find a good basin
        # Use 50 calls for random sampling
        n_random = min(50, budget // 10)
        for i in range(n_random):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best starting point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        x = best_x.copy()
        sigma = 2.0
        # CMA-ES parameters
        c1 = 1 / (dim + 3)
        cmu = 2 / ((dim + 2) ** 2 + 5)
        cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Population size
        lambda_ = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        
        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        step_size = sigma
        
        # Precompute sqrt of C for sampling
        # We'll update C each generation using rank-one and rank-mu updates
        
        for gen in range(1000):  # max generations
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            
            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                # Sample from N(0, C)
                # Use Cholesky or eigen decomposition for sqrt(C)
                # For simplicity and speed in low dim, use eigen
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                z = rng.standard_normal(dim)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                xi = x + step_size * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                offspring.append((fi, xi))
            
            if len(offspring) == 0:
                break
                
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            for fi, xi in offspring:
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            # Take top mu
            selected = [offspring[i][1] for i in range(min(mu, len(offspring)))]
            selected_f = [offspring[i][0] for i in range(min(mu, len(offspring)))]
            
            # Compute mean of selected
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += weights[i] * selected[i]
            
            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
            step_size *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            step_size = np.clip(step_size, 1e-10, 100)
            
            # Update covariance
            v = (new_x - x) / step_size
            C = (1 - c1 - cmu) * C + c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T) + \
                cmu * sum(weights[i] * np.outer(selected[i] - x, (selected[i] - x)) / step_size ** 2 for i in range(mu))
            
            # Update pc
            pc = (1 - cc) * pc + cc * hsig * v
            
            x = new_x
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
