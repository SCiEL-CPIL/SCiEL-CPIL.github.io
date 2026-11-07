import numpy as np
import time

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
        deadline = self.start_time + 280.0

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

        # Phase 1: Coarse search with Nelder-Mead
        rng = np.random.default_rng(42)
        start_points = [x0.copy()]
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, d))

        def nelder_mead(x_start, f_start):
            nonlocal best_f, best_x
            if not time_check():
                return

            # Simplex initialization
            delta = 1.0
            simplex = [x_start.copy()]
            f_vals = [f_start]

            for i in range(d):
                x_i = x_start.copy()
                x_i[i] += delta
                f_i = safe_eval(x_i)
                update_best(x_i, f_i)
                simplex.append(x_i)
                f_vals.append(f_i)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            tol = 1e-10

            max_iters = 150
            for _ in range(max_iters):
                if not time_check() or self.calls >= N:
                    break

                # Sort simplex
                combined = list(zip(f_vals, simplex))
                combined.sort(key=lambda x: x[0])
                f_vals = [c[0] for c in combined]
                simplex = [c[1] for c in combined]

                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                x_best = simplex[0]
                f_best = f_vals[0]

                if abs(f_worst - f_best) < tol:
                    break

                centroid = np.sum(simplex[:-1], axis=0) / d

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = safe_eval(x_r)
                update_best(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = safe_eval(x_e)
                    update_best(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    f_c = safe_eval(x_c)
                    update_best(x_c, f_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_vals[i] = safe_eval(simplex[i])
                            update_best(simplex[i], f_vals[i])

        for sp in start_points:
            if not time_check() or self.calls >= N:
                break
            f_sp = safe_eval(sp)
            update_best(sp, f_sp)
            nelder_mead(sp, f_sp)

        # Phase 2: Coordinate Descent with Golden Section Search
        # Bent Cigar has a narrow valley. Coordinate descent is effective.
        
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

            # Dynamic range coordinate descent
            # Start with a moderate range, shrink if converging, expand if stuck
            range_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            max_sweeps = 30
            for sweep in range(max_sweeps):
                if not time_check() or self.calls >= N:
                    break
                
                # Choose range size based on sweep number
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

                    new_val, new_f = golden_section_search(x_current, i, x_lo, x_hi, max_evals=12)
                    
                    if new_f < best_f:
                        x_current[i] = new_val
                        improved = True
                    else:
                        # If no improvement, keep current value
                        pass
                
                if not improved:
                    # If no improvement in this sweep, break if range is already small
                    if r_size <= 0.01:
                        break
                    # Otherwise continue to next sweep with smaller range

        return best_x, best_f
