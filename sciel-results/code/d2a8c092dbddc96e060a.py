# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start coordinate-wise line search for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        n = self.dim
        budget = self.budget

        state = {
            'calls': 0,
            'best_f': np.inf,
            'best_x': np.zeros(n)
        }

        def wrapped_func(x):
            if state['calls'] >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        def golden_section_1d(f, x, axis, lo, hi, tol=1e-8, max_iter=50):
            """Golden-section search along one axis."""
            if state['calls'] >= budget or time.time() > deadline:
                return x, f(x)
            
            phi = (1 + np.sqrt(5)) / 2
            invphi = 1.0 / phi
            invphi2 = 1.0 / (phi * phi)
            
            a = lo
            b = hi
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            
            x_c = x.copy()
            x_d = x.copy()
            x_c[axis] = c
            x_d[axis] = d
            
            fc = wrapped_func(x_c)
            fd = wrapped_func(x_d)
            
            if fc == np.inf or fd == np.inf:
                return x, f(x)
            
            for _ in range(max_iter):
                if time.time() > deadline or state['calls'] >= budget - 2:
                    break
                if b - a < tol:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - invphi * (b - a)
                    x_c = x.copy()
                    x_c[axis] = c
                    fc = wrapped_func(x_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    x_d = x.copy()
                    x_d[axis] = d
                    fd = wrapped_func(x_d)
                if fc == np.inf or fd == np.inf:
                    break
            
            # Pick the better of c and d
            if fc < fd:
                x_new = x.copy()
                x_new[axis] = c
                return x_new, fc
            else:
                x_new = x.copy()
                x_new[axis] = d
                return x_new, fd

        def coordinate_descent(x_init, max_rounds=20):
            """Coordinate-wise line search."""
            x = x_init.copy()
            fx = wrapped_func(x)
            if fx == np.inf:
                return x, fx
            
            for _ in range(max_rounds):
                if time.time() > deadline or state['calls'] >= budget - 2:
                    break
                improved = False
                for axis in range(n):
                    if time.time() > deadline or state['calls'] >= budget - 2:
                        break
                    # Determine search range around current value
                    cur_val = x[axis]
                    half_range = 2.0
                    lo = max(-5.0, cur_val - half_range)
                    hi = min(5.0, cur_val + half_range)
                    
                    if hi - lo < 1e-8:
                        continue
                    
                    x_new, fx_new = golden_section_1d(wrapped_func, x, axis, lo, hi)
                    
                    if fx_new < fx - 1e-10:
                        x = x_new
                        fx = fx_new
                        improved = True
                        # Shrink range for next iteration
                        half_range *= 0.5
                    else:
                        # Try a wider range if no improvement
                        half_range = min(5.0, half_range * 1.5)
                        lo = max(-5.0, cur_val - half_range)
                        hi = min(5.0, cur_val + half_range)
                        if hi - lo > 1e-8:
                            x_new, fx_new = golden_section_1d(wrapped_func, x, axis, lo, hi)
                            if fx_new < fx - 1e-10:
                                x = x_new
                                fx = fx_new
                                improved = True
            
            return x, fx

        # Multi-start with deterministic grid + random points
        rng = np.random.default_rng(42)
        
        # Generate initial points: center, corners, and random points
        init_points = []
        
        # Center
        init_points.append(np.zeros(n))
        
        # Axis-aligned points
        for i in range(n):
            for val in [-2.0, 0.0, 2.0]:
                p = np.zeros(n)
                p[i] = val
                init_points.append(p)
        
        # Random points
        for _ in range(15):
            init_points.append(rng.uniform(-4, 4, n))
        
        # Sort by distance to center (prefer closer points for early progress)
        init_points.sort(key=lambda p: np.linalg.norm(p))
        
        best_x = np.zeros(n)
        best_f = np.inf
        
        for x_init in init_points:
            if time.time() > deadline or state['calls'] >= budget - 10:
                break
            
            x_opt, f_opt = coordinate_descent(x_init, max_rounds=15)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        # Final polish with more rounds on the best point
        if state['calls'] < budget - 5 and time.time() < deadline:
            x_final, f_final = coordinate_descent(best_x, max_rounds=30)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
