import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 290.0

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        # Track best globally
        best_x = np.zeros(d)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        # Phase 1: Multi-start L-BFGS-B optimization
        # Bent Cigar is smooth and convex, so gradient-based methods converge fast
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        start_points = [x0.copy()]
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, d))
        
        # Add some points along axes for the narrow valley structure
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = sign * 3.0
                start_points.append(pt)
        
        # Run L-BFGS-B from each start point
        for sp in start_points:
            if not time_check() or self.calls >= N:
                break
            
            # Evaluate the start point
            f_sp = safe_eval(sp)
            update_best(sp, f_sp)
            
            if np.isinf(f_sp):
                continue
            
            # Run L-BFGS-B
            def objective(x):
                return safe_eval(x)
            
            try:
                result = minimize(
                    objective,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * d,
                    options={'maxiter': 200, 'maxfun': N - self.calls, 'maxls': 20}
                )
                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        # Phase 2: Fine-tune with coordinate descent using golden section search
        if self.calls < N - 10 and time_check():
            x_current = best_x.copy()
            
            def golden_section_search(x_base, coord_idx, x_lo, x_hi, max_evals=15):
                nonlocal best_f, best_x
                if not time_check() or self.calls >= N - max_evals:
                    return x_base[coord_idx], best_f
                
                inv_phi = (1 + np.sqrt(5)) / 2
                inv_phi = 1 / inv_phi
                
                c = x_hi - inv_phi * (x_hi - x_lo)
                dd = x_lo + inv_phi * (x_hi - x_lo)
                
                x_c = x_base.copy()
                x_c[coord_idx] = c
                f_c = safe_eval(x_c)
                update_best(x_c, f_c)
                
                x_dd = x_base.copy()
                x_dd[coord_idx] = dd
                f_dd = safe_eval(x_dd)
                update_best(x_dd, f_dd)
                
                for _ in range(max_evals - 2):
                    if not time_check() or self.calls >= N:
                        break
                    
                    if f_c < f_dd:
                        x_hi = dd
                        dd = c
                        f_dd = f_c
                        c = x_hi - inv_phi * (x_hi - x_lo)
                        x_c = x_base.copy()
                        x_c[coord_idx] = c
                        f_c = safe_eval(x_c)
                        update_best(x_c, f_c)
                    else:
                        x_lo = c
                        c = dd
                        f_c = f_dd
                        dd = x_lo + inv_phi * (x_hi - x_lo)
                        x_dd = x_base.copy()
                        x_dd[coord_idx] = dd
                        f_dd = safe_eval(x_dd)
                        update_best(x_dd, f_dd)
                
                if f_c < f_dd:
                    return c, f_c
                else:
                    return dd, f_dd

            # Dynamic range coordinate descent for final refinement
            range_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            max_sweeps = 20
            for sweep in range(max_sweeps):
                if not time_check() or self.calls >= N:
                    break
                
                if sweep < len(range_sizes):
                    r_size = range_sizes[sweep]
                else:
                    r_size = range_sizes[-1]
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    x_lo = max(x_current[i] - r_size, -5.0)
                    x_hi = min(x_current[i] + r_size, 5.0)
                    
                    if x_hi - x_lo < 1e-8:
                        continue

                    new_val, new_f = golden_section_search(x_current, i, x_lo, x_hi, max_evals=10)
                    
                    if new_f < best_f:
                        x_current[i] = new_val
                        improved = True
                
                if not improved:
                    if r_size <= 0.005:
                        break

        return best_x, best_f
