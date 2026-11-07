"""Rewritten program using LHS + custom finite-difference BFGS strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    """Hybrid LHS and custom finite-difference BFGS optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Deadline guard: keep 10s margin (300s limit)
        deadline = start_time + 290

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        # 1. Generate LHS points in [-4, 4]^dim
        # Using 30 points for global exploration
        n_lhs = 30
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_lhs)
        # Scale to [-4, 4] to ensure we are well within bounds for local solvers
        lhs_points = lhs_points * 8.0 - 4.0

        # Evaluate LHS points
        lhs_results = []
        for x in lhs_points:
            if call_count[0] >= budget - 150: # Reserve budget for local
                break
            f_val = wrapped_func(x)
            lhs_results.append((f_val, x))
            
        if not lhs_results:
             # Fallback
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
        else:
            lhs_results.sort(key=lambda item: item[0])
            best_f, best_x = lhs_results[0]
            
            # Take top 4 distinct points
            top_k = 4
            starts = [p[1] for p in lhs_results[:top_k]]
            
            bounds = [(-5.0, 5.0)] * dim
            
            for x0 in starts:
                if time.time() > deadline:
                    break
                if call_count[0] >= budget - 20:
                    break
                    
                # 2. Custom Finite-Difference BFGS
                try:
                    x = np.array(x0, dtype=float)
                    f = wrapped_func(x)
                    
                    # Initialize BFGS Hessian approximation
                    H = np.eye(dim)
                    
                    # Finite difference gradient estimator
                    def finite_diff_grad(x):
                        grad = np.zeros(dim)
                        h = 1e-5
                        for i in range(dim):
                            xp = x.copy()
                            xp[i] += h
                            # Clip to bounds
                            xp = np.clip(xp, -5.0, 5.0)
                            fp = wrapped_func(xp)
                            
                            xm = x.copy()
                            xm[i] -= h
                            xm = np.clip(xm, -5.0, 5.0)
                            fm = wrapped_func(xm)
                            
                            grad[i] = (fp - fm) / (2 * h)
                        return grad

                    g = finite_diff_grad(x)
                    
                    # BFGS Loop
                    max_iter = 100
                    for _ in range(max_iter):
                        if time.time() > deadline:
                            break
                        if call_count[0] >= budget - 10:
                            break
                        
                        # Line search
                        step = -H @ g
                        alpha = 1.0
                        f_new = f
                        x_new = x
                        
                        # Simple backtracking line search
                        for _ in range(20):
                            x_trial = x + alpha * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            f_trial = wrapped_func(x_trial)
                            
                            if f_trial < f - 1e-4 * alpha * (g @ step):
                                x_new = x_trial
                                f_new = f_trial
                                break
                            alpha *= 0.5
                        
                        if not np.allclose(x_new, x):
                            g_new = finite_diff_grad(x_new)
                            s = x_new - x
                            y = g_new - g
                            
                            if y @ s > 1e-12:
                                # Update Hessian approximation
                                Hy = H @ y
                                rho = 1.0 / (y @ s)
                                H = H + rho * (s @ s) * (1 + rho * (y @ Hy)) - rho * (s @ Hy + Hy @ s)
                                
                                x = x_new
                                f = f_new
                                g = g_new
                                
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                        else:
                            break

                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
