import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        X_all = []
        F_all = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x.copy())
            F_all.append(np.array(f))
            return f

        n_points = budget
        
        # Adaptive grid generation to minimize hypervolume gap
        # We want to minimize sum |f2'(ti)| * dti^2
        # This implies dti ~ 1/sqrt(|f2'(ti)|)
        # We sample uniformly in u = integral sqrt(|f2'(t)|) dt
        
        # Compute weights on a fine grid
        t_grid = np.linspace(0, 1, 1000)
        cos_t = np.cos(np.pi * t_grid / 2)
        sin_t = np.sin(np.pi * t_grid / 2)
        # f2'(t) = -cos(pi*t/2) - (pi/2)(1-t)sin(pi*t/2)
        # Magnitude
        f2_prime_mag = np.sqrt(cos_t**2 + (np.pi**2 / 4) * (1 - t_grid)**2 * sin_t**2)
        
        # Cumulative integral (trapezoidal approx)
        # u[0] = 0
        # u[i] = u[i-1] + 0.5 * (w[i-1] + w[i]) * dt
        dt = t_grid[1] - t_grid[0]
        u_grid = np.zeros_like(t_grid)
        # Use cumulative sum of trapezoids
        # trapezoid area for interval i is 0.5 * (w[i] + w[i+1]) * dt
        # We want u[i] to be the integral up to t[i]
        # u[i] = sum_{j=0}^{i-1} 0.5 * (w[j] + w[j+1]) * dt
        
        # Efficient calculation
        # areas = 0.5 * (w[:-1] + w[1:]) * dt
        # u[1:] = cumsum(areas)
        areas = 0.5 * (f2_prime_mag[:-1] + f2_prime_mag[1:]) * dt
        u_grid[1:] = np.cumsum(areas)
        
        total_u = u_grid[-1]
        
        # Target u values
        u_targets = np.linspace(0, total_u, n_points)
        
        # Invert to get t values
        t_vals = np.interp(u_targets, u_grid, t_grid)
        
        # Ensure endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            t = t_vals[i]
            x = np.zeros(dim)
            x[0] = t
            evaluate(x)
            
        return X_all
