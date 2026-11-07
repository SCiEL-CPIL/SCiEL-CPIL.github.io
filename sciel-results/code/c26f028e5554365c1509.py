"""Optimizer for BBOB f6 (Attractive sector) using CMA-ES with global exploration."""
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for f6 with global-to-local strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_x = x_clipped.copy()
                best_f = f
            return f

        def time_left():
            return deadline - time.time()

        # Phase 1: Global exploration with 50 uniform random evaluations
        rng = np.random.default_rng(42)
        init_points = []
        for _ in range(50):
            x = rng.uniform(-4.0, 4.0, dim)
            f = eval_func(x)
            init_points.append((f, x))
        
        if calls >= budget or time_left() < 5.0:
            return best_x, best_f

        # Sort by fitness and pick the best starting point
        init_points.sort(key=lambda x: x[0])
        x0 = init_points[0][1].copy()
        
        # Phase 2: CMA-ES optimization from the best point
        # Implement CMA-ES manually for full control over budget and time
        
        # CMA-ES parameters
        popsize = max(4, int(4 + 3 * np.log(dim)))  # lambda = 4 + 3*ln(n)
        popsize = min(popsize, 15)  # Cap population size for 5D
        
        # Initial step size: based on the spread of initial points
        # Use a moderate initial sigma
        sigma = 1.0
        
        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute Cholesky decomposition of C
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        mu = popsize // 2  # Number of best individuals used for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step sizes for paths
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Damping for covariance matrix update
        d_corr = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Remaining budget for CMA-ES
        remaining = budget - calls
        
        # Run CMA-ES
        for gen in range(200):  # Max generations
            if calls >= budget or time_left() < 2.0:
                break
            
            # Check if we can afford a full generation
            if calls + popsize > budget:
                # Do a partial generation
                n_evals = budget - calls
                if n_evals < 2:
                    break
            else:
                n_evals = popsize
            
            # Sample population
            samples = []
            for i in range(n_evals):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform using Cholesky
                try:
                    y = C_chol @ z
                except Exception:
                    y = z
                x_sample = mean + sigma * y
                samples.append(x_sample)
            
            # Evaluate population
            fvals = []
            for x_sample in samples:
                f = eval_func(x_sample)
                fvals.append(f)
                if calls >= budget or time_left() < 2.0:
                    break
            
            if len(fvals) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            idx = idx[:mu]  # Take best mu
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * np.array([samples[i] for i in idx]), axis=0)
            
            # Update evolution paths
            # First, compute the weighted sum of selected solutions
            y_best = (new_mean - mean) / sigma
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_best
            ps_norm = np.linalg.norm(ps)
            
            pc = (1 - c_c) * pc + c_c * y_best
            
            # Update covariance matrix
            # C = (1 + mu_eff * c_sigma * (2 - c_sigma) - c_c * (1 - c_c)) * C
            #   + c_sigma * (2 - c_sigma) * ps * ps.T
            #   + c_c * (1 - c_c) * pc * pc.T
            #   + (1 - c_sigma) * sum(weights_i * y_i * y_i.T)
            
            # Compute the outer product terms
            y_selected = np.array([samples[i] - mean for i in idx]) / sigma
            
            # Rank-1 update for ps
            update1 = c_sigma * (2 - c_sigma) * np.outer(ps, ps)
            
            # Rank-1 update for pc
            update2 = c_c * (1 - c_c) * np.outer(pc, pc)
            
            # Rank-mu update
            update3 = np.zeros((dim, dim))
            for i in range(mu):
                yi = y_selected[i]
                update3 += weights[i] * np.outer(yi, yi)
            
            # Combine
            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (d_corr - 1)
            cmu = 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff)
            
            C = (1 + c1 + cmu - c_sigma * (2 - c_sigma) - c_c * (1 - c_c)) * C + update1 + update2 + cmu * update3
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Ensure positive semi-definite by adding small regularization
            C += 1e-10 * np.eye(dim)
            
            # Recompute Cholesky
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity
                C = np.eye(dim)
                C_chol = np.eye(dim)
            
            # Update sigma using CSA (Cumulative Step Adaptation)
            h_sigma = min(1, (np.sqrt((1 - (1 - c_sigma) ** (2 * (gen + 1))) / 
                          (1 - (1 - c_sigma) ** 2)) - 1 + 1e-8) / 
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            sigma *= np.exp(c_sigma / chi_n * (ps_norm / chi_n - 1))
            
            # Limit sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            mean = new_mean
            
            # Check for stagnation: if sigma is very small or no improvement
            if sigma < 1e-10:
                break
            
            # Stagnation detection: if best fitness hasn't improved in last 30 evaluations
            # We'll track this more carefully below
        
        return best_x, best_f
