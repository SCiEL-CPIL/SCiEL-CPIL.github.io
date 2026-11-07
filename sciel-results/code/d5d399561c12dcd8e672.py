import numpy as np
import time


class Optimizer:
    """Derivative-free trust-region optimizer for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 305.0
        n = self.dim
        B = self.budget

        def time_ok():
            return (time.time() - start) < deadline

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def ev(x):
            nonlocal calls, best_x, best_f
            calls += 1
            y = np.clip(x, -5, 5)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return y, f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse sampling to find a good starting point
        # Sample corners and midpoints of the box
        coarse_points = []
        for i in range(2):
            for j in range(2):
                pt = np.zeros(n)
                pt[0] = -4.0 + i * 8.0
                pt[1] = -4.0 + j * 8.0
                coarse_points.append(pt)
        for i in range(n):
            for val in [-3.0, 0.0, 3.0]:
                pt = np.zeros(n)
                pt[i] = val
                coarse_points.append(pt)

        for pt in coarse_points:
            if calls >= B or not time_ok():
                break
            ev(pt)

        # Phase 2: DFTR (Derivative-Free Trust Region) optimization
        # Use a set of directional probes to build a local quadratic model
        
        def dftr_iterate(x_center, delta, max_iter=200):
            """One DFTR iteration: build quadratic model, optimize, update trust region."""
            nonlocal calls, best_x, best_f
            
            # Build stencil: center + 2n axis points
            stencil = [x_center.copy()]
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                stencil.append(x_center + delta * e)
                stencil.append(x_center - delta * e)
            
            # Evaluate stencil
            fvals = []
            for pt in stencil:
                if calls >= B or not time_ok():
                    return None, None, None
                y, f = ev(pt)
                fvals.append(f)
            
            if len(fvals) < 2 * n + 1:
                return None, None, None
            
            # Build quadratic model from stencil values
            # f(x) ≈ f0 + g^T (x - c) + 0.5 (x-c)^T H (x-c)
            # From axis points:
            # f(c ± d*e_i) = f0 ± g_i*d + 0.5*H_ii*d^2
            # So: g_i = (f(c+d*e_i) - f(c-d*e_i)) / (2*d)
            #     H_ii = (f(c+d*e_i) + f(c-d*e_i) - 2*f0) / d^2
            
            f0 = fvals[0]
            g = np.zeros(n)
            H = np.zeros((n, n))
            
            for i in range(n):
                f_plus = fvals[2 * i + 1]
                f_minus = fvals[2 * i + 2]
                g[i] = (f_plus - f_minus) / (2 * delta)
                H[i, i] = (f_plus + f_minus - 2 * f0) / (delta ** 2)
            
            # Add off-diagonal Hessian terms using cross points if budget allows
            # For simplicity, use diagonal Hessian only (separable model)
            # This is less accurate but faster and more robust
            
            # Solve for minimizer of quadratic model:
            # 0 = g + H * (x_new - x_center)
            # x_new = x_center - H^{-1} g (if H is positive definite)
            
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) > 1e-8:
                # Positive definite: unique minimizer
                x_new = x_center - np.linalg.solve(H, g)
            else:
                # Use gradient descent step
                norm_g = np.linalg.norm(g)
                if norm_g > 1e-10:
                    x_new = x_center - 0.5 * delta * g / norm_g
                else:
                    x_new = x_center.copy()
            
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            if calls >= B or not time_ok():
                return x_new, f0, delta
            
            y_new, f_new = ev(x_new)
            
            # Check improvement
            actual = f_new - f0
            # Predicted improvement (for diagonal H):
            # pred = -0.5 * g^T H^{-1} g
            if np.min(eigvals) > 1e-8:
                pred = -0.5 * g @ np.linalg.solve(H, g)
            else:
                pred = -0.5 * np.linalg.norm(g) * delta
            
            ratio = actual / max(abs(pred), 1e-10) if abs(pred) > 1e-10 else (1.0 if actual < 0 else 0.0)
            
            # Update trust region
            if ratio > 0.75:
                delta_new = min(2.0 * delta, 5.0)
            elif ratio > 0.25:
                delta_new = delta
            else:
                delta_new = 0.5 * delta
            
            return x_new, f_new, delta_new

        # Main DFTR loop
        x = best_x.copy()
        delta = 1.0
        no_improve_count = 0
        
        for it in range(300):
            if calls >= B or not time_ok():
                break
            
            x_new, f_new, delta_new = dftr_iterate(x, delta)
            
            if x_new is None:
                break
            
            if f_new < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            x = x_new
            delta = delta_new
            
            if delta < 1e-4:
                # Restart with larger delta
                delta = 0.5
                # Random restart around best
                if rng.random() < 0.3:
                    x = best_x + rng.standard_normal(n) * 0.5
                    x = np.clip(x, -5, 5)
            
            if no_improve_count > 30:
                # Restart
                no_improve_count = 0
                delta = 1.0
                x = best_x + rng.standard_normal(n) * 0.8
                x = np.clip(x, -5, 5)

        # Phase 3: Local polish with smaller steps
        if calls < B and time_ok():
            x = best_x.copy()
            delta = 0.1
            
            for it in range(100):
                if calls >= B or not time_ok():
                    break
                
                x_new, f_new, delta_new = dftr_iterate(x, delta)
                
                if x_new is None:
                    break
                
                x = x_new
                delta = delta_new
                
                if delta < 1e-5:
                    break

        return best_x, best_f
