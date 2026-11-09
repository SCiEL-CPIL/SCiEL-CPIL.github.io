"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Powell + Nelder-Mead optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 330s limit, leave 40s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = []
        points.append(np.zeros(d))
        for corner in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            pts = []
            for i in range(d):
                if i % 2 == 0:
                    pts.append(corner[0] * 3.0)
                else:
                    pts.append(corner[1] * 3.0)
            points.append(np.array(pts))
        
        for i in range(d):
            x = np.zeros(d)
            x[i] = 2.5
            points.append(x.copy())
            x[i] = -2.5
            points.append(x.copy())

        n_random = max(1, min(15, budget - len(points)))
        for _ in range(n_random):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Powell method for direct search along axes
        def powell_search(x0, max_calls):
            """Run Powell's method starting from x0."""
            nonlocal best_x, best_fx, calls_made
            
            def wrapped_func(x):
                if not time_ok():
                    return np.inf
                return call_func(np.clip(x, -5, 5))
            
            options = {
                'maxiter': max(1, max_calls - calls_made),
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            try:
                result = minimize(wrapped_func, x0, method='Powell', options=options)
                if result.fun < best_fx and np.isfinite(result.fun):
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
            
            return best_x, best_fx

        # Phase 3: Nelder-Mead local optimization
        def nelder_mead(x0, max_calls):
            """Run Nelder-Mead starting from x0."""
            nonlocal best_x, best_fx, calls_made
            
            def wrapped_func(x):
                if not time_ok():
                    return np.inf
                return call_func(np.clip(x, -5, 5))
            
            options = {
                'maxiter': max(1, max_calls - calls_made),
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            try:
                result = minimize(wrapped_func, x0, method='Nelder-Mead', options=options)
                if result.fun < best_fx and np.isfinite(result.fun):
                    best_fx = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
            
            return best_x, best_fx

        # Phase 4: Run Powell from the best point found in Phase 1
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                powell_calls = min(remaining, 200)
                powell_search(best_x.copy(), powell_calls)
        
        # Phase 5: Run Nelder-Mead from the best point
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                nm_calls = min(remaining, 150)
                nelder_mead(best_x.copy(), nm_calls)
        
        # Phase 6: Try Powell from a few other promising points
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 30:
                for trial in range(3):
                    if calls_made < budget and time.time() < deadline:
                        remaining = budget - calls_made
                        if remaining < 20:
                            break
                        x0 = best_x + rng.standard_normal(d) * 0.5
                        x0 = np.clip(x0, -5, 5)
                        fx0 = call_func(x0)
                        if fx0 < best_fx:
                            best_fx = fx0
                            best_x = x0.copy()
                        
                        if calls_made < budget and time.time() < deadline:
                            remaining = budget - calls_made
                            if remaining > 20:
                                powell_calls = min(remaining, 100)
                                powell_search(best_x.copy(), powell_calls)
        
        # Phase 7: Final polish with very small steps
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(30, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
