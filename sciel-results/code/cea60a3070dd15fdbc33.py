import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        d = self.dim
        N = self.budget
        time_limit = 280.0  # 300s limit, keep 20s margin

        def safe_func(x):
            self.calls += 1
            elapsed = time.time() - self.start_time
            if elapsed > time_limit:
                raise TimeoutError()
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Random sampling for global exploration
        n_global = min(30, N)
        points = np.random.uniform(-5, 5, (n_global, d))
        for i in range(n_global):
            if self.calls >= N or time.time() - self.start_time > time_limit:
                break
            try:
                safe_func(points[i])
            except TimeoutError:
                break

        if self.best_x is None:
            return self.best_x, self.best_f

        # Phase 2: Multi-start Nelder-Mead with adaptive step sizes
        # Try multiple restarts from best points found
        best_points = []
        # Collect some diverse starting points
        best_points.append(self.best_x.copy())
        
        # Add a few more diverse points near the best
        for _ in range(min(5, max(0, N - self.calls - 10))):
            if time.time() - self.start_time > time_limit:
                break
            # Perturb the best point
            perturb = np.random.uniform(-1, 1, d) * 0.5
            x_new = np.clip(self.best_x + perturb, -5, 5)
            try:
                fx = safe_func(x_new)
            except TimeoutError:
                break
            if fx < self.best_f + (self.best_f - self.best_f) * 0.1:  # near best
                best_points.append(x_new.copy())

        # For each starting point, run Nelder-Mead
        for start_idx, x0 in enumerate(best_points):
            if self.calls >= N or time.time() - self.start_time > time_limit:
                break
            
            # Adaptive step size based on how many evaluations we've used
            remaining = N - self.calls
            if remaining < d + 1:
                break
                
            # Step size: start moderate, decrease for later restarts
            base_step = 1.0 / (start_idx + 1)
            base_step = max(0.05, min(1.0, base_step))
            
            # Initialize simplex
            simplex = np.zeros((d + 1, d))
            simplex[0] = x0.copy()
            for i in range(1, d + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += base_step
            
            # Evaluate initial simplex
            simplex_f = np.zeros(d + 1)
            try:
                simplex_f[0] = self.best_f if np.array_equal(simplex[0], self.best_x) else safe_func(simplex[0])
            except TimeoutError:
                break
            
            for i in range(1, d + 1):
                if self.calls >= N or time.time() - self.start_time > time_limit:
                    break
                try:
                    simplex_f[i] = safe_func(simplex[i])
                except TimeoutError:
                    break
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            ftol = 1e-10
            xtol = 1e-8
            max_iter = min(200, max(10, (N - self.calls) // (d + 2)))
            
            for _ in range(max_iter):
                if self.calls >= N or time.time() - self.start_time > time_limit:
                    break
                
                # Sort simplex
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]
                
                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < ftol:
                    break
                if np.max(np.abs(simplex - simplex[0])) < xtol:
                    break
                
                # Centroid of all but worst
                x_bar = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_bar + alpha * (x_bar - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if self.calls >= N or time.time() - self.start_time > time_limit:
                    break
                try:
                    f_r = safe_func(x_r)
                except TimeoutError:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = x_bar + gamma * (x_r - x_bar)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if self.calls >= N or time.time() - self.start_time > time_limit:
                        if f_r < simplex_f[-1]:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        break
                    try:
                        f_e = safe_func(x_e)
                    except TimeoutError:
                        if f_r < simplex_f[-1]:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        x_c = x_bar + rho * (x_r - x_bar)
                    else:
                        x_c = x_bar - rho * (simplex[-1] - x_bar)
                    
                    x_c = np.clip(x_c, -5, 5)
                    
                    if self.calls >= N or time.time() - self.start_time > time_limit:
                        break
                    try:
                        f_c = safe_func(x_c)
                    except TimeoutError:
                        break
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if self.calls >= N or time.time() - self.start_time > time_limit:
                                break
                            try:
                                simplex_f[i] = safe_func(simplex[i])
                            except TimeoutError:
                                break
                        if self.calls >= N or time.time() - self.start_time > time_limit:
                            break

        # Phase 3: Final polish with scipy minimize (Nelder-Mead) if budget remains
        if self.calls < N and self.best_x is not None and time.time() - self.start_time < time_limit:
            try:
                result = minimize(
                    safe_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(100, N - self.calls),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = result.x.copy()
            except (TimeoutError, Exception):
                pass

        return self.best_x, self.best_f
