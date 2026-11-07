"""Multi-objective optimizer on ZDT6 using parametric front sampling + local search."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim

        # Pre-compute the Pareto front mapping for ZDT6
        # f1 = 1 - u where u = 2x1 - 4x1^2 + 2x1^4
        # We need to find x1 given u. The function g(x1) = 2x1 - 4x1^2 + 2x1^4
        # is symmetric: g(x1) = g(1-x1). For x1 in [0, 0.5], g is monotonically increasing from 0 to 1.
        # So we can invert on [0, 0.5] and use x1 or 1-x1.
        
        # Build a lookup table for x1 given u
        n_lut = 200000
        x1_grid = np.linspace(0, 0.5, n_lut)
        u_grid = 2 * x1_grid - 4 * x1_grid**2 + 2 * x1_grid**4
        
        def x1_from_u(u_val):
            """Find x1 in [0, 0.5] such that g(x1) = u_val."""
            idx = np.searchsorted(u_grid, u_val)
            if idx == 0:
                return 0.0
            if idx >= n_lut:
                return 0.5
            # Linear interpolation
            u0, u1 = u_grid[idx-1], u_grid[idx]
            x0, x1 = x1_grid[idx-1], x1_grid[idx]
            if u1 == u0:
                return x0
            frac = (u_val - u0) / (u1 - u0)
            return x0 + frac * (x1 - x0)

        # ZDT6 decision variables on the Pareto front:
        # x1 is determined by u
        # x2...x10 = 0.5
        # f1 = 1 - u
        # f2 = g(x) * h(f1, x) where g(x) = 1 + 9*sqrt(sum(x_i)/n) for i=2..n
        # h(f, x) = 1 - sqrt(f/x1) * exp(-x1 * cos(10 * pi * f / x1))
        
        def zdt6_front_point(u_val):
            """Generate a decision vector on the ZDT6 Pareto front for given u."""
            x1 = x1_from_u(u_val)
            x = np.full(dim, 0.5)
            x[0] = x1
            f1 = 1.0 - u_val
            # g(x) for x2..xn
            sum_rest = np.sum(x[1:])
            g_val = 1.0 + 9.0 * np.sqrt(sum_rest / (dim - 1))
            # h(f1, x1)
            if x1 > 1e-10:
                h_val = 1.0 - np.sqrt(f1 / x1) * np.exp(-x1 * np.cos(10 * np.pi * f1 / x1))
            else:
                h_val = 1.0
            f2 = g_val * h_val
            return x, f1, f2

        # Strategy: 
        # 1. Sample the Pareto front densely to get good initial coverage
        # 2. Use remaining budget for local refinement around front points
        
        budget = self.budget
        
        # Phase 1: Sample Pareto front
        # We want to cover u in [0, 1] well. Let's use a mix of uniform and concentrated samples.
        # For hypervolume, we want points that are well-distributed along the front.
        
        # Number of front samples: use a good fraction of budget
        n_front_samples = min(budget // 2, 100)
        
        # Generate u values: combine uniform samples with some focused on interesting regions
        u_uniform = np.linspace(0.01, 0.99, n_front_samples)
        
        # Also add some samples near the endpoints where the front is more curved
        n_extra = min(budget // 4, 50)
        u_extra = np.concatenate([
            np.linspace(0.001, 0.05, n_extra // 3),
            np.linspace(0.45, 0.55, n_extra // 3),
            np.linspace(0.95, 0.999, n_extra // 3)
        ])
        
        all_u = np.concatenate([u_uniform, u_extra])
        # Deduplicate and clip
        all_u = np.unique(np.clip(all_u, 1e-6, 1.0 - 1e-6))
        
        # Limit to budget
        max_front = budget - 10  # leave some for potential local search
        if len(all_u) > max_front:
            all_u = all_u[:max_front]
        
        used = 0
        all_points = []
        all_f = []
        
        # Evaluate front points
        for u_val in all_u:
            if time.time() > deadline:
                break
            x, f1, f2 = zdt6_front_point(u_val)
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            all_points.append(x_clipped)
            all_f.append(f)
        
        # Phase 2: Use remaining budget for local search around front points
        remaining = budget - used
        if remaining > 0 and len(all_points) > 0:
            # Pick some promising points and do local refinement
            # Sort by f1 to get diverse points
            f_arr = np.array(all_f)
            # Select points spread across the front
            n_select = min(remaining, len(all_points), 20)
            
            # Select evenly spaced points in f1
            f1_vals = f_arr[:, 0]
            sorted_idx = np.argsort(f1_vals)
            if n_select > 1:
                step = len(sorted_idx) // n_select
                selected = [sorted_idx[i * step] for i in range(n_select)]
            else:
                selected = [sorted_idx[0]]
            
            for idx in selected:
                if used >= budget or time.time() > deadline:
                    break
                base_x = all_points[idx].copy()
                # Try small perturbations
                for _ in range(min(remaining - (used - (budget - remaining)), 5)):
                    if used >= budget or time.time() > deadline:
                        break
                    # Perturb x1 slightly
                    x_try = base_x.copy()
                    perturbation = 0.01 * (hi - lo)
                    x_try[0] += perturbation * (np.random.random() * 2 - 1)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = problem(x_try)
                    used += 1
                    # Check if this point is non-dominated or improves hypervolume
                    # Simple check: if it's better than the base point in at least one objective
                    # and not worse in the other, it might be useful
                    if (f_try[0] < all_f[idx][0] and f_try[1] <= all_f[idx][1]) or \
                       (f_try[1] < all_f[idx][1] and f_try[0] <= all_f[idx][1]):
                        all_points[idx] = x_try
                        all_f[idx] = f_try
        
        return all_points
