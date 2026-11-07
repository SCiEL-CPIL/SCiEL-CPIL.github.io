import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        D = self.dim
        # Population size for CMA-ES
        # Standard: lambda = 4 + floor(3 * ln(D))
        # For D=5, lambda = 9. Let's use 10 for simplicity or 8 for more generations.
        # With 500 budget, lambda=10 -> 50 generations.
        # lambda=5 -> 100 generations.
        # Let's use lambda = 2*D = 10.
        lam = 10
        
        # Initial parameters
        # Start at 0, assuming the shift is not too large (it's in [-4,4])
        # Or start at a random point? The task says minimizer is in [-4,4]^5.
        # Starting at 0 is a good guess.
        x_mean = np.zeros(D)
        sigma = 2.0  # Initial step size
        
        # Weights
        mu = lam // 2  # Number of weights
        # Standard weights: log weights
        weights = np.zeros(lam)
        for i in range(mu):
            weights[i] = np.log(lam / 2 + 0.5) - np.log(i + 1)
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (D + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (D + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        c_1 = 2 / ((D + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2)**2 + mu_eff))
        
        # State variables
        C = np.eye(D)
        C_sqrt = np.eye(D)  # Cholesky factor or just sqrt for simplicity (we need inverse sqrt for sampling)
        # Actually, we need to sample from N(0, C). 
        # Standard CMA-ES uses a matrix square root.
        # For simplicity and speed, I might use a simpler covariance update or just diagonal adaptation?
        # No, full CMA-ES is better.
        
        # To avoid complex Cholesky updates, I can use the standard algorithm.
        # But implementing Cholesky update is tricky.
        # Alternative: Use a library? No, "Write class Optimizer".
        # Maybe a simpler algorithm is safer.
        
        # Let's reconsider Nelder-Mead. It's much simpler to implement correctly.
        # And for D=5, it's very effective.
        
        # Let's go with a robust Nelder-Mead with restarts.
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def nm_simplex(x0, f0, sigma0):
            nonlocal calls, best_x, best_f
            # Initialize simplex
            # Vertices: x0 and x0 + sigma0 * e_i
            simplex = np.zeros((D + 1, D))
            simplex[0] = x0
            fvals = np.zeros(D + 1)
            fvals[0] = f0
            calls += 1
            
            for i in range(1, D + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += sigma0
                fvals[i] = func(simplex[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrinkage
            
            for _ in range(100):  # Max iterations per NM run
                if time.time() > deadline:
                    return
                
                # Sort
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                # Check convergence
                if fvals[-1] - fvals[0] < 1e-12:
                    break
                if np.max(np.abs(simplex - simplex[0])) < 1e-12:
                    break
                    
                # Centroid
                centroid = np.mean(simplex[1:], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, D + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if calls >= self.budget:
                            return

        # Strategy:
        # 1. Try NM from 0.
        # 2. If it doesn't improve enough or stalls, try from random points.
        
        # First attempt from 0
        x0 = np.zeros(D)
        f0 = func(x0)
        calls += 1
        best_f = f0
        best_x = x0.copy()
        
        nm_simplex(x0, f0, 1.0)
        
        # If we have budget left, try a few random restarts
        rng = np.random.default_rng(42)
        while calls < self.budget - (D + 1) and time.time() < deadline:
            x_start = rng.uniform(-4, 4, D)
            f_start = func(x_start)
            calls += 1
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            # Only run NM if it looks promising or if we have lots of budget
            # For efficiency, just run NM if we have enough budget for one full run
            if calls + 50 < self.budget:
                nm_simplex(x_start, f_start, 0.5)
            else:
                break
                
        return best_x, best_f
