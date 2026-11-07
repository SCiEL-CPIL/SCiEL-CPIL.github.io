"""Rewritten optimizer using a gradient-free BFGS quasi-Newton approach with finite-difference gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """BFGS quasi-Newton optimizer for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Initial exploration to find a good basin
        n_initial = min(20, budget // 10)
        for i in range(n_initial):
            if time.time() > deadline or calls >= budget:
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = np.ones(n) * 0.5
            elif i == 2:
                x = np.ones(n)
            elif i == 3:
                x = -np.ones(n)
            elif i == 4:
                x = np.array([1, 1, 1, 1, 1], dtype=float)
            elif i == 5:
                x = np.array([-1, -1, -1, -1, -1], dtype=float)
            elif i == 6:
                x = rng.uniform(-3, 3, n)
            elif i == 7:
                x = rng.uniform(-3, 3, n)
            elif i == 8:
                x = rng.uniform(-2, 2, n)
            else:
                x = rng.uniform(-4, 4, n)
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: BFGS quasi-Newton with finite-difference gradients
        # We'll implement a gradient-free BFGS that approximates the Hessian
        # from finite-difference gradient estimates.
        
        def estimate_gradient(x):
            """Estimate gradient using central finite differences."""
            nonlocal calls
            if calls >= budget:
                return None
            grad = np.zeros(n)
            h = 1e-5
            for j in range(n):
                if calls >= budget:
                    return None
                x_plus = x.copy()
                x_plus[j] += h
                x_minus = x.copy()
                x_minus[j] -= h
                f_plus = safe_eval(x_plus)
                if calls >= budget:
                    return None
                f_minus = safe_eval(x_minus)
                if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                    return None
                grad[j] = (f_plus - f_minus) / (2 * h)
            return grad

        def line_search(x, direction, f_current, max_iter=50):
            """Backtracking line search."""
            nonlocal calls
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            grad = estimate_gradient(x)
            if grad is None:
                return alpha, x
            slope = np.dot(grad, direction)
            if slope >= 0:
                return 0, x
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                x_new = x + alpha * direction
                f_new = safe_eval(x_new)
                if not np.isfinite(f_new):
                    alpha *= rho
                    continue
                if f_new <= f_current + c1 * alpha * slope:
                    return alpha, x_new
                alpha *= rho
            return alpha, x

        # BFGS implementation
        x_current = best_x.copy()
        f_current = best_fx
        
        # Initial Hessian approximation
        H = np.eye(n)
        max_bfgs_iters = min(30, (budget - calls) // (2 * n + 5))
        
        for it in range(max_bfgs_iters):
            if time.time() > deadline or calls >= budget:
                break
            
            # Estimate gradient
            grad = estimate_gradient(x_current)
            if grad is None:
                break
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-10:
                break
            
            # Search direction: p = -H @ grad
            p = -H @ grad
            
            # Line search
            alpha, x_new = line_search(x_current, p, f_current)
            
            if alpha <= 0 or not np.all(np.isfinite(x_new)):
                break
            
            f_new = safe_eval(x_new)
            if not np.isfinite(f_new):
                break
            
            # Update BFGS Hessian approximation
            grad_new = estimate_gradient(x_new)
            if grad_new is None:
                break
            
            s = x_new - x_current
            y = grad_new - grad
            sy = np.dot(s, y)
            
            if sy > 1e-12:
                Hy = H @ y
                H = H - np.outer(s, s) / sy + (np.outer(Hy, Hy)) / (np.dot(y, Hy))
            
            x_current = x_new
            f_current = f_new
        
        # Phase 3: If budget remains, do additional local refinement
        if best_x is not None and time.time() < deadline and calls < budget:
            # Try a few more BFGS runs from slightly perturbed points
            for attempt in range(3):
                if time.time() > deadline or calls >= budget:
                    break
                if budget - calls < 2 * n + 5:
                    break
                
                perturb = rng.standard_normal(n) * 0.05
                x_start = np.clip(best_x + perturb, -5, 5)
                f_start = safe_eval(x_start)
                
                if not np.isfinite(f_start):
                    continue
                
                # Run BFGS from this point
                x_curr = x_start.copy()
                f_curr = f_start
                H = np.eye(n)
                max_iters = min(15, (budget - calls) // (2 * n + 5))
                
                for it in range(max_iters):
                    if time.time() > deadline or calls >= budget:
                        break
                    
                    grad = estimate_gradient(x_curr)
                    if grad is None:
                        break
                    
                    if np.linalg.norm(grad) < 1e-10:
                        break
                    
                    p = -H @ grad
                    alpha, x_new = line_search(x_curr, p, f_curr)
                    
                    if alpha <= 0 or not np.all(np.isfinite(x_new)):
                        break
                    
                    f_new = safe_eval(x_new)
                    if not np.isfinite(f_new):
                        break
                    
                    grad_new = estimate_gradient(x_new)
                    if grad_new is None:
                        break
                    
                    s = x_new - x_curr
                    y = grad_new - grad
                    sy = np.dot(s, y)
                    
                    if sy > 1e-12:
                        Hy = H @ y
                        H = H - np.outer(s, s) / sy + (np.outer(Hy, Hy)) / (np.dot(y, Hy))
                    
                    x_curr = x_new
                    f_curr = f_new
        
        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
