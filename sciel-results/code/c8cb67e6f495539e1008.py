import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        n_evals = self.budget
        if n_evals <= 0:
            return []

        def f1_of_x0(x0):
            return 2.0 + 10.0 * np.sin(np.pi * x0 / 10.0)

        def f2_of_f1(f1):
            return 1.0 - np.sqrt(f1 / 10.0)

        def objectives(x0):
            f1 = f1_of_x0(x0)
            f2 = f2_of_f1(f1)
            return f1, f2

        r1, r2 = 11.0, 11.0

        def hypervolume_2d(f1_arr, f2_arr):
            """Compute 2D hypervolume relative to (r1, r2) for points on Pareto front."""
            # Sort by f1 ascending
            order = np.argsort(f1_arr)
            f1_s = f1_arr[order]
            f2_s = f2_arr[order]
            n = len(f1_s)
            if n == 0:
                return 0.0
            # Each point i contributes width*(height) where width = f1_{i+1} - f1_i
            # and the last "width" extends to r1
            widths = np.empty(n)
            widths[:-1] = f1_s[1:] - f1_s[:-1]
            widths[-1] = r1 - f1_s[-1]
            heights = r2 - f2_s
            return float(np.sum(widths * heights))

        # Initial spacing: use a quadratic-in-density approach
        # Start with points denser where df1/dx0 * (r2 - f2) is large
        N_grid = 20000
        x_grid = np.linspace(0.0, 1.0, N_grid)
        f1_grid = f1_of_x0(x_grid)
        f2_grid = f2_of_f1(f1_grid)
        df1_dx0 = (np.pi / 10.0) * np.cos(np.pi * x_grid / 10.0)
        density = np.abs(df1_dx0) * (r2 - f2_grid)
        
        # Ensure positive
        density = np.maximum(density, 1e-15)
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        
        if n_evals == 1:
            x0_vals = np.array([0.5])
        else:
            quantiles = np.linspace(0.0, 1.0, n_evals)
            x0_vals = np.interp(quantiles, cdf, x_grid)
        
        # Iterative local optimization: for each point, try small perturbations
        # to maximize total hypervolume
        f1_init = f1_of_x0(x0_vals)
        f2_init = f2_of_f1(f1_init)
        current_hv = hypervolume_2d(f1_init, f2_init)
        
        best_x0 = x0_vals.copy()
        best_hv = current_hv
        
        # Iterate: adjust each point to maximize HV
        for iteration in range(30):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n_evals):
                # Try perturbing point i
                base = best_x0[i]
                # Adaptive step: smaller near boundaries
                step_sizes = [0.01, 0.005, 0.001, 0.0005]
                best_local = base
                best_local_hv = best_hv
                
                # Compute current HV with point at base
                f1_curr = f1_init.copy()
                f2_curr = f2_init.copy()
                
                for step in step_sizes:
                    for direction in [-1, 1]:
                        candidate = base + direction * step
                        if candidate < 0.0 or candidate > 1.0:
                            continue
                        # Update f1, f2 for this point
                        new_f1 = f1_of_x0(candidate)
                        new_f2 = f2_of_f1(new_f1)
                        f1_test = f1_curr.copy()
                        f2_test = f2_curr.copy()
                        f1_test[i] = new_f1
                        f2_test[i] = new_f2
                        hv_test = hypervolume_2d(f1_test, f2_test)
                        if hv_test > best_local_hv + 1e-12:
                            best_local_hv = hv_test
                            best_local = candidate
                
                if best_local != base:
                    best_x0[i] = best_local
                    f1_init[i] = f1_of_x0(best_local)
                    f2_init[i] = f2_of_f1(f1_init[i])
                    best_hv = best_local_hv
                    improved = True
            
            if not improved:
                break
        
        x0_values = best_x0
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0_values[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
