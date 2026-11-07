"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restart for Lunacek bi-Rastrigin (f24)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        fbest_global = np.inf
        calls_total = 0
        
        # Strategy: Multiple CMA-ES runs with different starting points and sigmas
        # For bi-Rastrigin, we need to explore both basins
        # Run 1: Large sigma from origin (explores broadly)
        # Run 2: Medium sigma from a kicked point
        # Run 3+: Smaller sigma from best found or another kick
        
        # Define a schedule of (start_offset, sigma_init, budget_fraction)
        # We'll use 500 total calls
        
        # First run: broad exploration
        run1_budget = min(200, budget)
        x0_1, sigma_1, best_x_1, fbest_1, calls_used_1 = self._run_cmaes(
            func, run1_budget, dim,
            start_point=np.zeros(dim),
            sigma_init=3.0
        )
        calls_total += calls_used_1
        if fbest_1 < fbest_global:
            fbest_global = fbest_1
            best_x_global = best_x_1
        
        # Second run: kick from best or random
        if calls_total < budget - 20 and time.time() < deadline - 15:
            remaining = budget - calls_total
            run2_budget = min(150, remaining)
            # Try a kick in a random direction
            kick_dir = np.random.standard_normal(dim)
            kick_dir /= np.linalg.norm(kick_dir)
            # For bi-Rastrigin, the second basin is often opposite to the first
            start2 = np.clip(-2.0 * best_x_global + 1.0 * kick_dir, -5, 5)
            x0_2, sigma_2, best_x_2, fbest_2, calls_used_2 = self._run_cmaes(
                func, run2_budget, dim,
                start_point=start2,
                sigma_init=2.0
            )
            calls_total += calls_used_2
            if fbest_2 < fbest_global:
                fbest_global = fbest_2
                best_x_global = best_x_2
        
        # Third run: if budget remains, try another kick
        if calls_total < budget - 20 and time.time() < deadline - 10:
            remaining = budget - calls_total
            run3_budget = min(100, remaining)
            kick_dir = np.random.standard_normal(dim)
            kick_dir /= np.linalg.norm(kick_dir)
            start3 = np.clip(best_x_global + 3.0 * kick_dir, -5, 5)
            x0_3, sigma_3, best_x_3, fbest_3, calls_used_3 = self._run_cmaes(
                func, run3_budget, dim,
                start_point=start3,
                sigma_init=1.5
            )
            calls_total += calls_used_3
            if fbest_3 < fbest_global:
                fbest_global = fbest_3
                best_x_global = best_x_3
        
        # Fourth run: polish around the best found
        if calls_total < budget - 10 and time.time() < deadline - 5 and best_x_global is not None:
            remaining = budget - calls_total
            run4_budget = min(50, remaining)
            x0_4, sigma_4, best_x_4, fbest_4, calls_used_4 = self._run_cmaes(
                func, run4_budget, dim,
                start_point=best_x_global.copy(),
                sigma_init=0.5
            )
            calls_total += calls_used_4
            if fbest_4 < fbest_global:
                fbest_global = fbest_4
                best_x_global = best_x_4
        
        # Final local search: coordinate descent with decreasing step
        if calls_total < budget and time.time() < deadline - 2 and best_x_global is not None:
            best_x_global, fbest_global = self._local_search(func, best_x_global, fbest_global, dim, budget - calls_total)
        
        return best_x_global, fbest_global

    def _run_cmaes(self, func, budget, dim, start_point=None, sigma_init=1.0):
        """Run CMA-ES with given budget and parameters."""
        if budget < 20:
            # Too few calls, just evaluate start point
            if start_point is None:
                start_point = np.zeros(dim)
            f = func(start_point)
            return start_point, sigma_init, start_point, f, 1
        
        # CMA-ES parameters for dim=5
        lambda_ = min(15, max(10, dim + 5))  # 10 for dim=5
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        if start_point is None:
            x0 = np.zeros(dim)
        else:
            x0 = start_point.copy()
            
        sigma = sigma_init
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute B = C^{1/2}
        def sqrtm(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                return vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        
        B = sqrtm(C)
        
        fbest = func(x0)
        best_x = x0.copy()
        calls = 1
        
        # E[||ps||] for h_sigma
        E_norm_ps = np.sqrt(dim * (1 - (1 - c_sigma)**2))
        
        while calls < budget:
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                y = x0 + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                
                fy = func(y)
                fs.append(fy)
                calls += 1
                
                if fy < fbest:
                    fbest = fy
                    best_x = y.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            fs = np.array(fs)
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = fs[order]
            
            # Update mean
            x_new = sum(w[i] * ys[i] for i in range(mu))
            
            # Compute C^{-1/2}
            vals, vecs = np.linalg.eigh(C)
            vals_inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-12))
            C_inv_sqrt = (vecs * vals_inv_sqrt) @ vecs.T
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x0) / sigma
            
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < 1.4 * E_norm_ps else 0.0
            sigma *= np.exp(c_sigma / dim * (np.linalg.norm(ps) / E_norm_ps - 1) * h_sigma)
            
            # Update covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x0) / sigma
            yk = x_new - x0
            rank1 = np.outer(yk, yk)
            rankmu = sum(w[i] * np.outer(ys[i] - x0, ys[i] - x0) for i in range(mu))
            
            C = (1 + c_1 * (c_mu * mu_eff / dim - 1) - c_1 - c_mu) * C \
                + c_1 * (pc @ pc + (1 - (1 - c_c)**2) * C) \
                + c_mu * rankmu
            
            C = (C + C.T) / 2
            
            x0 = x_new
            B = sqrtm(C)
            
            # Convergence check
            if sigma < 1e-8:
                break
            if calls > 50 and sigma < 1e-6:
                break
        
        return x0, sigma, best_x, fbest, calls

    def _local_search(self, func, x_init, f_init, dim, remaining_budget):
        """Simple coordinate descent local search."""
        if remaining_budget < 10:
            return x_init, f_init
        
        best_x = x_init.copy()
        best_f = f_init
        calls_made = 0
        
        step = 0.3
        for _ in range(remaining_budget):
            improved = False
            for d in range(dim):
                for sign in [1, -1]:
                    if calls_made >= remaining_budget:
                        break
                    x_trial = best_x.copy()
                    x_trial[d] += sign * step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls_made += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial
                        improved = True
                if calls_made >= remaining_budget:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-7:
                    break
            elif calls_made >= remaining_budget:
                break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
