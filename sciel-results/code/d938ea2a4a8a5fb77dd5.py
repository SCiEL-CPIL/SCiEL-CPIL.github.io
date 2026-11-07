import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES and return (best_x, best_f, calls_used, converged)"""
        d = self.dim
        lam = 4 + int(3 * np.log(d))
        
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mu = int(lam / 2)
        w = weights[:mu]
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 - mueff + c_sigma) / (d + mueff + 9)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        x_mean = x0.copy()
        sigma = sigma0
        
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Track convergence
        no_improve_generations = 0
        max_no_improve = 15
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            remaining = budget - calls
            if remaining < lam:
                break
            
            ys = [B @ (D * rng.standard_normal(d)) for _ in range(lam)]
            
            xs = []
            fs = []
            for y in ys:
                x_new = x_mean + sigma * y
                f_new = func(x_new)
                xs.append(x_new)
                fs.append(f_new)
                calls += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_generations = 0
            
            if len(fs) < lam:
                break

            idx = np.argsort(fs)
            xs_sorted = np.array(xs)[idx]
            fs_sorted = np.array(fs)[idx]
            
            # Check for stagnation
            if fs_sorted[0] >= best_f - 1e-10:
                no_improve_generations += 1
            else:
                no_improve_generations = 0
            
            if no_improve_generations >= max_no_improve:
                break
            
            x_mean_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            y_sigma = (x_mean_new - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            y_sorted = (xs_sorted[:mu] - x_mean) / sigma
            delta_h = (1 - h_sigma**2) * c_c * (2 - c_c)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + delta_h * C)
            C_new += c_mu * (y_sorted.T @ (w[:, None] * y_sorted))
            
            C_new = (C_new + C_new.T) / 2
            
            eigenvalues, B_new = np.linalg.eigh(C_new)
            eigenvalues = np.maximum(eigenvalues, 1e-16)
            D_new = np.sqrt(eigenvalues)
            
            x_mean = x_mean_new
            C = C_new
            B = B_new
            D = D_new
            
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            if sigma < 1e-6:
                break
        
        converged = (no_improve_generations >= max_no_improve) or (sigma < 1e-5)
        return best_x, best_f, calls, converged

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Phase 1: Multiple CMA-ES runs with adaptive restarts
        # Use ~70% of budget for global search with restarts
        global_budget = int(total_budget * 0.7)
        
        # Initial points: center + random points
        initial_points = [np.zeros(d)]
        for _ in range(3):
            initial_points.append(rng.uniform(-4, 4, d))
        
        for i, x_init in enumerate(initial_points):
            if calls_used >= global_budget or time.time() > deadline:
                break
            
            remaining_budget = global_budget - calls_used
            if remaining_budget < 30:  # Need at least some budget for a run
                break
            
            # Adaptive sigma based on distance from center
            dist_from_center = np.linalg.norm(x_init)
            sigma_init = max(1.5, min(3.0, 2.0 + 0.1 * dist_from_center))
            
            x_best, f_best, calls_this_run, converged = self._cma_es_run(
                func, x_init, sigma_init, remaining_budget, rng, deadline
            )
            
            calls_used += calls_this_run
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            # If converged and we have budget left, do another restart
            if not converged and calls_used < global_budget:
                continue
            
            # If we've used significant budget and found a good solution, stop
            if calls_used > global_budget * 0.5 and best_f < 50:
                break
        
        # Phase 2: Local polish using Nelder-Mead
        if best_x is not None and calls_used < total_budget:
            polish_budget = total_budget - calls_used
            
            # Use Nelder-Mead for local refinement
            # Limit function calls in polish phase
            max_polish_calls = min(polish_budget, 100)
            
            # Create a wrapper that tracks calls
            polish_calls = [0]
            polish_limit = max_polish_calls
            
            def wrapped_func(x):
                if polish_calls[0] >= polish_limit:
                    return np.inf
                polish_calls[0] += 1
                return func(x)
            
            try:
                result = minimize(
                    wrapped_func, best_x, method='Nelder-Mead',
                    options={'maxiter': max_polish_calls, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                calls_used += polish_calls[0]
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass
        
        # If we haven't made any calls (shouldn't happen), make one
        if calls_used == 0:
            x0 = np.zeros(d)
            f0 = func(x0)
            best_x = x0
            best_f = f0
            calls_used = 1
        
        return best_x, best_f
