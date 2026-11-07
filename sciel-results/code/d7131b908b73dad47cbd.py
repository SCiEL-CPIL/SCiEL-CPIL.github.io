"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin: 300s limit, 10s buffer
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            # Ensure x is a 1D array of length d
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling to find a good starting point
        # Use a few random points and the origin
        init_points = [np.zeros(d)]
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, d))
        
        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = try_eval(best_x)
            
        if calls_used >= max_calls or time.time() > deadline - 5:
            return best_x, best_f

        # Phase 2: CMA-ES
        # Parameters
        pop_size = 4 + int(3 * np.log(d))  # Standard pop size formula
        if pop_size < 4: pop_size = 4
        # For d=5, pop_size = 4 + 3*1.6 = ~8.8 -> 8
        
        lambda_ = pop_size
        mu = lambda_ // 2  # Selection pressure
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Restart criteria
        max_restarts = 10
        min_sigma = 1e-6
        max_sigma = 10.0
        
        for restart in range(max_restarts):
            if calls_used >= max_calls or time.time() > deadline - 3:
                break
            
            # Initialize CMA-ES state around current best
            mean = best_x.copy()
            sigma = 1.0
            C = np.eye(d)
            Cinv = np.eye(d) # Not strictly needed if we use Cholesky, but good for reference
            
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            # Precompute Cholesky factor of C (initially I)
            B = np.eye(d)
            D = np.ones(d)
            
            # Main loop for this restart
            while calls_used < max_calls and time.time() < deadline - 3:
                # Generate population
                ys = []
                for i in range(lambda_):
                    z = rng.standard_normal(d)
                    # x = mean + sigma * B @ (D * z)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                
                # Evaluate population
                fvals = []
                for y in ys:
                    fy = try_eval(y)
                    if fy is None:
                        break
                    fvals.append(fy)
                
                if calls_used >= max_calls or len(fvals) < lambda_:
                    break
                
                fvals = np.array(fvals)
                
                # Sort by fitness
                order = np.argsort(fvals)
                ys_sorted = [ys[i] for i in order]
                fvals_sorted = fvals[order]
                
                # Update best
                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = ys_sorted[0].copy()
                
                # Select best mu individuals
                y_selected = ys_sorted[:mu]
                
                # Update mean
                old_mean = mean.copy()
                mean = np.sum([w[i] * y_selected[i] for i in range(mu)], axis=0)
                mean = np.clip(mean, -5, 5)
                
                # Update evolution paths
                # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B^T * (mean - old_mean) / sigma
                y_diff = (mean - old_mean) / sigma if sigma > 0 else np.zeros(d)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_diff)
                
                # h_sigma indicator
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used % 100 + 1))) < (1 + 2 / (d + 1)) * chi_n
                
                # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * B^T * (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ y_diff)
                
                # Update covariance matrix
                # C = (1 - c_1 - c_mu + delta_hsig * c_1 * c_c * (2 - c_c)) * C
                #    + c_1 * (p_c * p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C)
                #    + c_mu * sum(w_i * (y_i - old_mean) * (y_i - old_mean)^T / sigma^2)
                
                delta_hsig = (1 - h_sigma) * c_c * (2 - c_c)
                
                # Rank-1 update term
                y_selected_diffs = np.array([(y - old_mean) / sigma for y in y_selected])
                
                C = (1 - c_1 - c_mu + delta_hsig * c_1 * c_c * (2 - c_c)) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_hsig * C)
                
                # Rank-mu update term
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(y_selected_diffs[i], y_selected_diffs[i])
                C += c_mu * rank_mu
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(min(sigma, max_sigma), min_sigma)
                
                # Check for convergence / stagnation
                # If sigma is very small or the population is very tight
                # We can check the condition number of C or the volume
                # For simplicity, if sigma < min_sigma * 10, restart
                if sigma < min_sigma * 10:
                    break
                    
                # Update Cholesky decomposition of C
                # C = B @ diag(D^2) @ B.T
                try:
                    L = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(L))
                    B = L / D[:, np.newaxis]
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset C
                    C = np.eye(d)
                    B = np.eye(d)
                    D = np.ones(d)
                    sigma = max(sigma, 0.1) # Increase sigma to escape
                
                # If the mean has not changed significantly, we might be stuck
                # But CMA-ES handles this via sigma adaptation.
                # We break if we detect no improvement in last few generations?
                # For now, rely on sigma shrinking or restart limit.
                
                # If the population is too small (std of fvals is tiny), restart
                if np.std(fvals_sorted) < 1e-8:
                    break

            # After inner loop, if we didn't use all budget, prepare for next restart
            if calls_used < max_calls and time.time() < deadline - 3:
                # Reset sigma to a larger value for the next restart
                # Start from the best point found so far
                # Keep mean as best_x
                pass # The loop above handles restarts by re-initializing state

        # Phase 3: Local polish (optional, but CMA-ES usually does this well)
        # If we have budget left, do a few (1+1)-ES steps
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(max_calls - calls_used):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
