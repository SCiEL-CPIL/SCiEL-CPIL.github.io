import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return float(func(x))

        def alive():
            return calls[0] < budget and time.time() < deadline

        rng = np.random.default_rng(12345)

        # ---- Phase 0: Deterministic grid initialization ----
        # BBOB f11 is shifted. The minimizer is in [-4, 4]^5.
        # A coarse grid helps locate the basin of attraction.
        # We use a grid of size 3^5 = 243 points? No, budget is 500.
        # Let's use a smaller set or adaptive.
        # The inspiration suggests "deterministic grid initialization".
        # Let's try a grid on [-4, 4] with step 2.0 -> -4, -2, 0, 2, 4 (5 points per dim)
        # 5^5 = 3125 points. Too many for 500 budget.
        # Let's try step 4.0 -> -4, 0, 4 (3 points per dim). 3^5 = 243 points.
        # This fits in the budget and covers the likely range.
        
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*([grid_vals] * d))).T.reshape(-1, d)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate grid points
        for p in grid_points:
            if not alive():
                break
            p = np.clip(p, -5, 5)
            fp = wrapped(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        if best_x is None:
            # Fallback if grid failed or budget exhausted
            best_x = np.zeros(d)
            best_f = wrapped(best_x)

        x = best_x.copy()
        fx = best_f

        # ---- Phase 1: Per-axis 1D scans to find descent direction and scale ----
        # This handles the rotated/anisotropic quadratic nature of f11.
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        
        for _ in range(3):  # A few passes to refine
            if not alive():
                break
            for axis in range(d):
                if not alive():
                    break
                e = np.zeros(d)
                e[axis] = 1.0
                found = False
                for s in step_sizes:
                    if not alive():
                        break
                    # Try both directions
                    for sign in (1.0, -1.0):
                        cand = np.clip(x + sign * s * e, -5, 5)
                        fc = wrapped(cand)
                        if fc < fx - 1e-12:
                            # Descent found
                            x = cand
                            fx = fc
                            found = True
                            break
                    if found:
                        break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # ---- Phase 2: Coordinate Descent with Golden-Section Line Search ----
        # For a quadratic function, coordinate descent with line search converges linearly.
        # Given the low dimension (5), this is efficient.
        
        phi = (np.sqrt(5.0) - 1.0) / 2.0  # Golden ratio

        def line_search(axis, x0, fx0, init_step):
            # 1D minimization over t in x0 + t*e_axis
            e = np.zeros(d)
            e[axis] = 1.0

            def g(t):
                pt = np.clip(x0 + t * e, -5, 5)
                return wrapped(pt)

            # Determine bracket
            # We know there is a descent direction from Phase 1 or previous iterations.
            # But we need a robust bracket.
            # Start with a small range and expand if necessary, or use the known descent step.
            
            # Simple approach: scan a small range around 0 to find a local minimum bracket.
            # Since we are doing coordinate descent, the function is unimodal along each axis
            # if we are in the convex region (which f11 is globally, being a quadratic).
            
            # Let's use a grid scan to find the approximate minimum, then refine.
            # Range: +/- init_step
            # Number of points: 11
            t_vals = np.linspace(-init_step, init_step, 11)
            
            best_t = 0.0
            best_g = fx0
            for t in t_vals:
                if not alive():
                    break
                gt = g(t)
                if gt < best_g:
                    best_g = gt
                    best_t = t
            
            if best_t == 0.0:
                # No improvement found in this range
                return x0, fx0, 0.0

            # Refine using Golden Section around the best grid point
            # Bracket: [best_t - step, best_t + step] roughly?
            # Better: [t_vals[idx-1], t_vals[idx+1]]
            
            # Find index of best_t in t_vals
            idx = np.argmin(np.abs(t_vals - best_t))
            
            if idx == 0:
                lo = t_vals[0]
                hi = t_vals[1]
            elif idx == len(t_vals) - 1:
                lo = t_vals[-2]
                hi = t_vals[-1]
            else:
                lo = t_vals[idx - 1]
                hi = t_vals[idx + 1]
                
            # Ensure lo < hi
            if lo > hi:
                lo, hi = hi, lo
                
            # Golden Section Search
            a = lo
            b = hi
            c = b - phi * (b - a)
            dpt = a + phi * (b - a)
            
            fc = g(c)
            fd = g(dpt)
            
            for _ in range(15):
                if not alive():
                    break
                if fc < fd:
                    b = dpt
                    dpt = c
                    fd = fc
                    c = b - phi * (b - a)
                    fc = g(c)
                else:
                    a = c
                    c = dpt
                    fc = fd
                    dpt = a + phi * (b - a)
                    fd = g(dpt)
                    
            t_mid = 0.5 * (a + b)
            fm = g(t_mid)
            
            if fm < best_g:
                best_g = fm
                best_t = t_mid
            
            pt = np.clip(x0 + best_t * e, -5, 5)
            return pt, best_g, best_t

        # Iterate coordinate descent
        # Start with a larger step, then shrink
        initial_step = 1.0
        
        for _ in range(20):  # Max outer iterations
            if not alive():
                break
            improved = False
            for axis in range(d):
                if not alive():
                    break
                x, fx, step = line_search(axis, x, fx, initial_step)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if abs(step) > 1e-6:
                    improved = True
            
            if not improved:
                # Try finer scale
                initial_step = 0.05
                fine_improved = False
                for axis in range(d):
                    if not alive():
                        break
                    x, fx, step = line_search(axis, x, fx, initial_step)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if abs(step) > 1e-7:
                        fine_improved = True
                if not fine_improved:
                    break

        return best_x, best_f
