"""Multi-start Powell with coarse basin mapping for BBOB f4 (Büche-Rastrigin)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        # ---------- Phase 0: Coarse grid sampling to map basin structure ----------
        # Büche-Rastrigin has periodic structure. Sample a coarse grid to find promising basins.
        # Use a 3x3x3x3x3 grid = 243 points, but we only have 500 budget total.
        # Let's use a 2x2x2x2x2 grid = 64 points first, then refine.
        # Actually, let's be smarter: sample 100 random points + some structured ones.
        
        # Start with center and some structured points
        init_pts = [np.zeros(dim)]
        
        # Add axis-aligned points
        for axis in range(dim):
            for s in (-3.0, -1.5, 1.5, 3.0):
                p = np.zeros(dim)
                p[axis] = s
                init_pts.append(p)
        
        # Add random points
        while len(init_pts) < 80:
            init_pts.append(rng.uniform(-4.5, 4.5, dim))
        
        # Evaluate initial points
        for p in init_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)
        
        # ---------- Phase 1: Multi-start Powell refinement ----------
        # Collect unique promising starting points from the initial evaluation
        # We'll use the best points found so far as starting points for Powell
        
        # Sort initial points by function value (we need to track them)
        # Let's re-evaluate: we stored best_x, best_f but not all points.
        # Let's do a fresh set of Powell starts based on random + best found.
        
        # Use the current best and several random points as Powell starts
        powell_starts = [best_x.copy()]
        
        # Add random starts
        num_powell_starts = 15
        for _ in range(num_powell_starts):
            if call_count >= budget - 20 or not time_ok():
                break
            p = rng.uniform(-4.5, 4.5, dim)
            p = np.clip(p, -4.95, 4.95)
            powell_starts.append(p)
        
        # Run Powell from each start
        for x0 in powell_starts:
            if call_count >= budget - 10 or not time_ok():
                break
            
            try:
                # Use L-BFGS-B or Powell
                # Powell is derivative-free and good for multimodal
                res = minimize(func, x0, method='Powell',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                # Budget exceeded or other error - continue with next start
                continue
        
        # ---------- Phase 2: Final fine-tune with local coordinate descent ----------
        if call_count < budget - 5 and time_ok():
            x = best_x.copy()
            f = best_f
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
            
            for step in step_sizes:
                if call_count >= budget - 2 or not time_ok():
                    break
                improved = True
                while improved and call_count < budget - 2 and time_ok():
                    improved = False
                    for i in range(dim):
                        if call_count >= budget - 2 or not time_ok():
                            break
                        base = x[i]
                        for delta in [step, -step]:
                            if call_count >= budget - 2 or not time_ok():
                                break
                            c = base + delta
                            if not (-4.95 <= c <= 4.95):
                                continue
                            x[i] = c
                            cf = do_call(x)
                            if cf < f:
                                f = cf
                                improved = True
                                best_f = f
                                best_x = x.copy()
                            else:
                                x[i] = base
        
        return best_x, best_f
