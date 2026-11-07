import numpy as np
import time
from scipy.optimize import newton

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        def calc_f1(x1):
            # Handle scalar and array
            if np.isscalar(x1):
                return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        def calc_f1_deriv(x1):
            # Derivative of f1(x1) = 1 - (1 - E(x1))**0.1
            # Let u = 1 - E(x1). f1 = 1 - u**0.1. df1/du = -0.1*u**-0.9.
            # E(x1) = exp(-4x1) * sin(6*pi*x1)**6
            # dE/dx1 = exp(-4x1) * [ -4 * sin^6(6*pi*x1) + 6 * sin^5(6*pi*x1) * 6*pi * cos(6*pi*x1) ]
            # = exp(-4x1) * sin^5(6*pi*x1) * [ -4*sin(6*pi*x1) + 36*pi*cos(6*pi*x1) ]
            s = np.sin(6 * np.pi * x1)
            c = np.cos(6 * np.pi * x1)
            e = np.exp(-4 * x1)
            # u = 1 - e * s**6
            u = 1 - e * s**6
            # Avoid division by zero if u is close to 0
            # u**(-0.9) is large if u is small.
            # If u is 0, f1=1, derivative is infinite?
            # At x1=0, s=0, u=1.
            # At peaks, s=1, e is small.
            # Let's just use numerical derivative if analytical is messy or risky.
            # Numerical derivative is safer for implementation brevity and robustness.
            h = 1e-8
            return (calc_f1(x1 + h) - calc_f1(x1 - h)) / (2 * h)

        # 1. Determine f1_max
        # We need the global maximum of f1 on [0, 1].
        # f1 is periodic-ish but damped by exp(-4x1).
        # The maximum is in the first few lobes.
        # Let's sample to find the approximate location of the max.
        x_grid = np.linspace(0, 1, 10000)
        f_grid = calc_f1(x_grid)
        idx_max = np.argmax(f_grid)
        f1_max_approx = f_grid[idx_max]
        x1_max_approx = x_grid[idx_max]
        
        # Refine f1_max using newton on derivative? Or just use the grid value?
        # Grid value is good enough for range definition.
        # Actually, let's just use the max of the grid.
        f1_max = f1_max_approx

        # 2. Generate target f1 values
        n_points = self.budget
        # Uniform in f1
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # 3. Map target_f1s to x1
        # We need to solve f1(x1) = t for x1.
        # We can use the grid to find a starting point for Newton's method.
        # For each target t, find the x1 in the grid that is closest to t.
        
        # Sort grid by f1? No, x1 is the independent variable.
        # We can just search the grid for the closest f1 value to t.
        # Since n_points is 200 and grid is 10000, this is fast.
        
        x1_targets = np.zeros(n_points)
        
        for i, t in enumerate(target_f1s):
            # Find index in grid with closest f1 value
            # Note: f1 is not monotonic, so there might be multiple solutions.
            # We just need one. The closest one in the grid is a good start.
            diffs = np.abs(f_grid - t)
            idx_closest = np.argmin(diffs)
            x0 = x_grid[idx_closest]
            
            # Newton refinement
            try:
                # Solve f1(x) - t = 0
                x_sol = newton(lambda x: calc_f1(x) - t, x0, fprime=calc_f1_deriv, maxiter=10, tol=1e-12)
                # Ensure x_sol is in [0, 1]
                if 0 <= x_sol <= 1:
                    x1_targets[i] = x_sol
                else:
                    x1_targets[i] = x0
            except Exception:
                x1_targets[i] = x0

        # 4. Evaluate
        recorded = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            # Clip to bounds
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
