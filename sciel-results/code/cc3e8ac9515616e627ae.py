import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for base in range(2, 2 + dim):
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    r += f * (i_temp % base)
                    i_temp //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def _cma_es_run(self, func, x0, sigma0, max_calls, rng, deadline, best_f_tracker):
        """
        Run CMA-ES.
        Returns: best_x, best_f, calls_used
        """
        d = self.dim
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
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
        
        no_improve_generations = 0
        max_no_improve = 10
        
        while calls < max_calls:
            if time.time() > deadline:
                break
            
            remaining = max_calls - calls
            if remaining < lam:
                break
            
            # Sample
            ys = rng.standard_normal((lam, d))
            xs = x_mean + sigma * (ys @ B.T * D)
            
            xs_list = []
            fs_list = []
            
            for i in range(lam):
                x_new = xs[i]
                f_new = func(x_new)
                xs_list.append(x_new)
                fs_list.append(f_new)
                calls += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_generations = 0
            
            if len(fs_list) < lam:
                break

            idx = np.argsort(fs_list)
            xs_sorted = np.array(xs_list)[idx]
            fs_sorted = np.array(fs_list)[idx]
            
            # Check stagnation
            if fs_sorted[0] >= best_f - 1e-10:
                no_improve_generations += 1
            else:
                no_improve_generations = 0
            
            if no_improve_generations >= max_no_improve:
                x_mean = best_x.copy()
                sigma = sigma * 1.5
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                no_improve_generations = 0
                if sigma > 10.0:
                    break
                continue

            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            y_sigma = (x_mean_new - x_mean) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update C
            y_sorted = (xs_sorted[:mu] - x_mean) / sigma
            delta_h = (1 - h_sigma**2) * c_c * (2 - c_c)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + delta_h * C)
            C_new += c_mu * (y_sorted.T @ (weights[:, None] * y_sorted))
            
            C_new = (C_new + C_new.T) / 2
            
            # Eigen decomposition
            eigenvalues, B_new = np.linalg.eigh(C_new)
            eigenvalues = np.maximum(eigenvalues, 1e-16)
            D_new = np.sqrt(eigenvalues)
            
            x_mean = x_mean_new
            C = C_new
            B = B_new
            D = D_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            if sigma < 1e-6:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Phase 1: Initial Sampling
        n_initial = 20
        halton_pts = self._halton_sequence(n_initial, d)
        
        special_points = [
            np.zeros(d),
            np.full(d, 2.0),
            np.full(d, -2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0] if d == 5 else [2.0] * d),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0] if d == 5 else [-2.0] * d),
        ]
        
        initial_candidates = []
        for pt in special_points:
            initial_candidates.append(pt)
        
        for pt in halton_pts:
            x = -4 + 8 * pt
            initial_candidates.append(x)
        
        for x in initial_candidates:
            if calls_used >= total_budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_used += 1
            
        # Phase 2: CMA-ES
        sigma_init = 1.0
        remaining_budget = total_budget - calls_used
        
        if remaining_budget > 30:
            x_best, f_best, calls_cma = self._cma_es_run(
                func, best_x, sigma_init, remaining_budget, rng, deadline, best_f
            )
            calls_used += calls_cma
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
        
        # Phase 3: Local Polish
        if best_x is not None and calls_used < total_budget:
            polish_budget = total_budget - calls_used
            max_polish_calls = min(polish_budget, 50)
            
            if max_polish_calls > 0:
                polish_calls = [0]
                polish_limit = max_polish_calls
                
                def wrapped_func(x):
                    if polish_calls[0] >= polish_limit:
                        return np.inf
                    polish_calls[0] += 1
                    return func(np.clip(x, -5, 5))
                
                try:
                    from scipy.optimize import minimize
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
        
        if calls_used == 0:
            x0 = np.zeros(d)
            f0 = func(x0)
            best_x = x0
            best_f = f0
            calls_used = 1
            
        return best_x, best_f
