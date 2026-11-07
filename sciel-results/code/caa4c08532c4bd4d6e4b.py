"""Optimizer for BBOB f9 (rotated Rosenbrock) in 5D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings("ignore")


class Optimizer:
    """
    Hybrid optimizer for rotated Rosenbrock in 5D.
    
    Strategy:
    1. Dense global sampling using a space-filling (Sobol-like via halton) sequence 
       to quickly locate the narrow valley of the Rosenbrock function.
    2. Differential Evolution for robust global search if the valley isn't found quickly.
    3. L-BFGS-B for high-precision local polishing once in the basin.
    4. Adaptive restarts and fine-tuning.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf
        self.evaluated_points = []  # List of (fx, x)

    def __call__(self, func):
        start_time = time.time()
        # Evaluator kills at 330s. We want to be safe.
        # Let's aim to finish well before 300s to be safe against overhead.
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            if np.isnan(fx):
                fx = np.inf

            self.evaluated_points.append((fx, x.copy()))
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            
            return fx

        # Helper to get best point
        def get_best():
            if self.best_x is None:
                return np.zeros(self.dim), np.inf
            return self.best_x, self.best_fx

        # Phase 1: Global Sampling
        # The Rosenbrock valley is narrow. We need good coverage.
        # Use a quasi-random sequence (Halton) for better space-filling than pure random.
        
        def halton_sequence(n, dim, skip=1):
            points = []
            for i in range(skip, skip + n):
                point = []
                for j in range(dim):
                    f = j + 2  # Base for dimension j (2, 3, 5, ...)
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        i_val, f_ = divmod(i_val, f)
                        r += f_ / (f ** (j + 1))
                    point.append(r)
                points.append(point)
            return np.array(points)

        # Generate ~100 Halton points in [0,1]^dim and map to [-5, 5]^dim
        # We have budget 500. Let's use 100 for initial global scan.
        n_halton = 80
        halton_pts = halton_sequence(n_halton, self.dim)
        # Map [0,1] to [-5,5]
        halton_pts = halton_pts * 10.0 - 5.0

        # Add origin and a few specific points
        special_pts = [np.zeros(self.dim)]
        # Maybe add points along axes?
        for i in range(self.dim):
            for val in [-4, -2, 0, 2, 4]:
                p = np.zeros(self.dim)
                p[i] = val
                special_pts.append(p)

        # Evaluate global points
        for x0 in halton_pts:
            if not time_ok() or self.call_count >= self.budget - 100:
                break
            try:
                tracked_func(x0)
            except Exception:
                break

        for x0 in special_pts:
            if not time_ok() or self.call_count >= self.budget - 100:
                break
            try:
                tracked_func(x0)
            except Exception:
                break

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Differential Evolution (Global Search)
        # DE is robust for global optimization. 
        # We have ~400 calls left.
        # DE with popsize=10, maxiter=50 -> 10*50 = 500 evals. Too many.
        # Let's use a smaller popsize and fewer iterations.
        
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 50 and time_ok():
            try:
                # Use DE with a limited budget
                # popsize * maxiter should be <= remaining_budget
                # Let's try popsize=10, maxiter=30 -> 300 evals
                # But we want to stop early if it finds something good.
                
                # DE options:
                # maxiter, popsize, tol, polish=True (uses L-BFGS-B at the end)
                # polish=True is great! It uses the remaining budget for polishing.
                
                bounds = [(-5.0, 5.0)] * self.dim
                
                # We need to be careful with the callback to enforce time/budget.
                # scipy DE doesn't have a simple callback to stop early based on external state easily 
                # without wrapping the func, which we did.
                # However, if tracked_func raises an exception, DE will crash.
                # So we must ensure we don't exceed budget inside DE.
                
                # Let's limit maxiter and popsize.
                # popsize=10, maxiter=20 -> 200 evals.
                # Plus polish might use more.
                
                # Let's try a smaller DE run first.
                de_res = differential_evolution(
                    tracked_func,
                    bounds=bounds,
                    seed=123,
                    maxiter=20,
                    popsize=10,
                    tol=1e-6,
                    polish=False, # We will polish ourselves to control budget
                    init='sobol'
                )
                
                if de_res.fun < self.best_fx:
                    self.best_fx = de_res.fun
                    self.best_x = de_res.x
            except Exception:
                pass

        # Phase 3: Local Optimization (L-BFGS-B)
        # Run L-BFGS-B from the current best point.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 20 and time_ok():
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': min(100, remaining_budget - 5),
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 4: Fine-tuning / Second Local Pass
        # If we have budget left, run another local optimization from the best point
        # or try a few random perturbations of the best point and optimize from them.
        
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 10 and time_ok():
            # Try 3 random starts near the best point
            for i in range(3):
                if not time_ok() or self.call_count >= self.budget - 10:
                    break
                perturbation = 0.5 * rng.standard_normal(self.dim)
                x_start = np.clip(self.best_x + perturbation, -5.0, 5.0)
                
                try:
                    res = minimize(
                        tracked_func,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={
                            'maxiter': 20,
                            'maxfun': min(20, remaining_budget - 5 - i*10),
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    break

        # Final fallback
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0

        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
