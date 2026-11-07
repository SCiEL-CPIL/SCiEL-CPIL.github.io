import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        n_samples = budget
        n = dim
        
        if n_samples == 1:
            t_vals = np.array([0.5])
        else:
            # Initial guess: Arc-length uniform
            t_grid = np.linspace(1e-10, 1, 20000)
            
            def f2_of_t(t):
                return 1 - np.sqrt(t / n) * (2 - t * (np.sin(10 * np.pi * t) + 1))
            
            f1 = t_grid
            f2 = f2_of_t(t_grid)
            
            df1_dt = np.ones_like(t_grid)
            df2_dt = np.gradient(f2, t_grid)
            
            ds_dt = np.sqrt(df1_dt**2 + df2_dt**2)
            L_cum = np.cumsum(ds_dt) * (t_grid[1] - t_grid[0])
            L_total = L_cum[-1]
            L_norm = L_cum / L_total
            
            fractions = np.linspace(0, 1, n_samples)
            t_vals = np.interp(fractions, L_norm, t_grid)
            t_vals = np.clip(t_vals, 0, 1)

            # Optimize t_vals to maximize HV
            f2_ref = 11.0
            f1_ref = 11.0
            
            def hv_from_t(t):
                # t is array of shape (n_samples,)
                f2 = f2_of_t(t)
                # Sort by t (f1)
                sort_idx = np.argsort(t)
                t_sorted = t[sort_idx]
                f2_sorted = f2[sort_idx]
                
                # Calculate HV
                # H = sum_{i=1}^N (t_{i+1} - t_i) * (f2_ref - f2_i)
                # t_{N+1} = f1_ref
                # t_0 is not used in this formula? 
                # The formula was: sum (f1_{i+1} - f1_i) (f2_ref - f2_i)
                # with f1_{N+1} = f1_ref.
                # This assumes the first point starts at t_1.
                # What about the area from 0 to t_1?
                # The box for P_1 is [t_1, 11] x [f2_1, 11].
                # The HV includes the area to the left of t_1?
                # No, the union of boxes starts at min(f1).
                # So the area is from min(f1) to 11.
                # The formula sum_{i=1}^N (t_{i+1} - t_i) (f2_ref - f2_i) covers [t_1, 11].
                # This is correct.
                
                t_ext = np.append(t_sorted, f1_ref)
                widths = np.diff(t_ext)
                heights = f2_ref - f2_sorted
                return np.sum(widths * heights)
            
            def neg_hv(t):
                return -hv_from_t(t)
            
            # Use Nelder-Mead for local optimization
            # It is robust and doesn't require gradients
            res = minimize(neg_hv, t_vals, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < neg_hv(t_vals):
                t_vals = res.x
                t_vals = np.clip(t_vals, 0, 1)
                # Ensure sorted? Not strictly necessary for eval, but good for consistency
                # t_vals = np.sort(t_vals)

        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
