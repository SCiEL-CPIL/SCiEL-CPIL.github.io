"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class Optimizer:
    """
    Optimizer for BBOB f9 (rotated Rosenbrock) in 5D.
    
    Strategy:
    1. Global sampling to find the basin of attraction.
    2. Local optimization using Powell's method (derivative-free, robust for valleys).
    3. Final polish with (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_fx = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Wrapper to track calls and enforce budget/deadline
        def tracked_func(x):
            if not time_ok() or self.call_count >= self.budget:
                raise ValueError("Budget or time exceeded")
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            self.call_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Initial Sampling
        # We want to cover the space well to find the basin.
        # 1. Origin
        # 2. Corners (2^5 = 32 points)
        # 3. Random points
        
        initial_points = []
        initial_points.append(np.zeros(self.dim))
        
        # Add corners
        for i in range(2**self.dim):
            corner = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            corner = corner * 10.0 - 5.0 # Map {0,1} to {-5, 5}
            initial_points.append(corner)
            
        # Add random points to fill gaps
        n_random = 10
        for _ in range(n_random):
            initial_points.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate initial points
        for x0 in initial_points:
            if not time_ok() or self.call_count >= self.budget - 50:
                break
            try:
                tracked_func(x0)
            except Exception:
                pass

        # If no valid points found, return zero
        if self.best_x is None:
             return np.zeros(self.dim), 0.0

        # Phase 2: Local Optimization using Powell
        # Powell is derivative-free and effective for Rosenbrock.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 20 and time_ok():
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfev': min(remaining_budget - 10, 300), 
                        'xtol': 1e-12, 
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 3: Refinement with (1+1)-ES
        # If we have budget left, do a fine-grained search around the best point.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 5 and time_ok() and self.best_x is not None:
            # Estimate step size
            sigma = 0.1
            x = self.best_x.copy()
            fx = self.best_fx
            
            # Simple local search to polish
            for _ in range(remaining_budget - 2):
                if not time_ok() or self.call_count >= self.budget - 1:
                    break
                
                # Gaussian step
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                
                try:
                    fy = tracked_func(y)
                except Exception:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05 # Slightly increase step on success
                else:
                    sigma *= 0.8  # Decrease on failure
                    if sigma < 1e-6:
                        break

        # Final check: if we have even more budget, run Powell again with tight bounds?
        remaining_budget = self.call_count # Wait, this is wrong.
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 10 and time_ok() and self.best_x is not None:
            try:
                # Use current best as start
                # Bounds: best_x +/- 0.5
                bounds = []
                for i in range(self.dim):
                    lo = max(-5.0, self.best_x[i] - 0.5)
                    hi = min(5.0, self.best_x[i] + 0.5)
                    bounds.append((lo, hi))
                    
                res = minimize(
                    tracked_func,
                    self.best_x,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': remaining_budget - 2,
                        'xtol': 1e-15,
                        'ftol': 1e-15
                    }
                )
            except Exception:
                pass

        # Ensure we return a valid solution
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_fx = 0.0
            
        return self.best_x, self.best_fx

# EVOLVE-BLOCK-END
