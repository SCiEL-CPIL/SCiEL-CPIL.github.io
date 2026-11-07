"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """Adaptive CMA-ES with multi-start restarting for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf
        global_best_x = None
        global_best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f, global_best_x, global_best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Use a mix of centered and uniform samples to quickly find a decent basin
        n_initial = 30
        candidates = []
        
        # Include zero and some structured points
        initial_points = [np.zeros(n)]
        for _ in range(n_initial - 1):
            # Mix of different scales to cover the domain
            scale = rng.choice([1.0, 2.0, 4.0])
            initial_points.append(rng.uniform(-scale, scale, n))
            
        for i, x in enumerate(initial_points):
            if calls >= budget or not time_ok():
                break
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            # Fallback if something went wrong
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return global_best_x, global_best_f

        # Sort candidates to find the best starting point for CMA-ES
        candidates.sort(key=lambda t: t[0])
        best_start_f, best_start_x = candidates[0]

        # Phase 2: Adaptive CMA-ES with Restarts
        # We will run CMA-ES, and if it stagnates, restart from the global best.
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(n))  # Standard population size
        mu = lambda_pop // 2  # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial step size based on the spread of initial candidates or a heuristic
        # If we have diverse candidates, use a larger sigma
        if len(candidates) > 1:
            # Estimate scale from top candidates
            top_xs = np.array([c[1] for c in candidates[:5]])
            spread = np.mean(np.std(top_xs, axis=0))
            sigma = max(0.5, spread * 0.5)
        else:
            sigma = 1.0
            
        sigma = min(sigma, 2.0) # Cap initial sigma

        # State variables
        mean = best_start_x.copy()
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        no_improve_count = 0
        max_no_improve = 30 # Restart if no improvement for this many iterations
        restarts = 0
        max_restarts = 10
        
        while calls < budget and time_ok() and restarts < max_restarts:
            # Generate population
            B = rng.standard_normal((lambda_pop, n))
            # Whitening transformation (Cholesky decomposition of C)
            # For 5D, this is cheap. We can use np.linalg.cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(n)
                L = np.eye(n)
            
            Y = B @ L
            P = mean + sigma * Y
            
            # Evaluate population
            fvals = np.full(lambda_pop, np.inf)
            for i in range(lambda_pop):
                if calls >= budget or not time_ok():
                    break
                fx = evaluate(P[i])
                if fx is not None:
                    fvals[i] = fx
            
            if np.all(np.isinf(fvals)):
                break # No valid evaluations, break loop
                
            # Sort by fitness
            order = np.argsort(fvals)
            P_sorted = P[order]
            fvals_sorted = fvals[order]
            
            # Update mean
            y = np.sum(weights[:, None] * P_sorted[:mu], axis=0)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * (y / (sigma * np.sqrt(np.diag(C)) if np.all(np.diag(C) > 1e-10) else y/sigma))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (y / sigma)
            
            # Delta-sigma rule
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # hsig = 1 if ||ps||^2 / (1 - (1-c_sigma)^{2(it+1)}) < (1 + 2/mu) * chi_n^2 else 0
            # Approximation for simplicity and speed in 5D
            ps_norm_sq = np.dot(ps, ps)
            hsig = 1.0 if ps_norm_sq < (1 + 2.0 / mueff) * chi_n**2 * (1 - (1 - c_sigma)**(2 * (calls - n_initial))) else 0.0
            
            # Rank-1 and rank-mu updates
            # y_sigma = y / sigma
            y_sigma = y / sigma
            C = (1 + c_1 * hsig * (c_c + 1 - c_c) - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) - (1 - hsig) * C)
            C += c_mu * np.sum(weights[:mu, None, None] * np.outer(P_sorted[:mu] - mean, P_sorted[:mu] - mean) / sigma**2, axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(n)
            
            # Update mean
            mean = mean + c_sigma * y # Wait, standard CMA-ES updates mean directly
            # Correction: The mean update is simply the weighted sum of the best points
            mean = np.sum(weights[:, None] * P_sorted[:mu], axis=0)
            
            # Check for improvement
            current_best_in_pop = fvals_sorted[0]
            if current_best_in_pop < global_best_f - 1e-12:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Restart condition
            if no_improve_count >= max_no_improve or sigma < 1e-8:
                restarts += 1
                # Restart from global best
                mean = global_best_x.copy()
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                # Reset sigma to a value that allows exploration
                # Heuristic: distance from global best to average of initial candidates?
                # Or just a fixed moderate value
                sigma = 1.0 
                no_improve_count = 0
                
        # Phase 3: Local Polish with (1+1)-ES
        # If we have budget left, do a quick local search to squeeze out the last bit
        if global_best_x is not None and calls < budget and time_ok():
            x = global_best_x.copy()
            f = global_best_f
            sigma = 0.1
            min_sigma = 1e-10
            max_sigma = 2.0
            no_improve = 0
            patience = 20

            for _ in range(budget - calls):
                if not time_ok() or calls >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.1, max_sigma)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.9, min_sigma)
                    no_improve += 1
                    
                    if no_improve >= patience and sigma < min_sigma * 10:
                        x = global_best_x.copy()
                        f = global_best_f
                        sigma = 0.5
                        no_improve = 0
                        patience = 10

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
