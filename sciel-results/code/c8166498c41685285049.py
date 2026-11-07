"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive (1+1)-ES with L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # State variables
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Start with a large sigma to cover the domain [-5, 5]
        x = rng.uniform(-4.5, 4.5, self.dim)
        fx = eval_func(x)
        sigma = 2.0
        
        # We will use about 60% of the budget for the ES phase
        es_budget = int(self.budget * 0.6)
        
        for _ in range(es_budget):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = eval_func(y)
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = np.array(y, copy=True)

            # 1/5 success rule adaptation
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1 # Increase step size on success
            else:
                sigma *= 0.9 # Decrease step size on failure
                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 2: L-BFGS-B Polish
        # Use the best point found by ES as the starting point for gradient-based optimization
        if best_x is not None and call_count < self.budget:
            polish_budget = self.budget - call_count
            # Allocate most of the remaining budget to L-BFGS-B
            # L-BFGS-B is very efficient, so we can allow many iterations
            # but we are limited by the total call budget.
            
            bounds = [(-5.0, 5.0)] * self.dim
            
            # We might try multiple starts if we have budget, but usually one good start from ES is enough.
            # Let's try 2 starts: the best point, and a slightly perturbed version.
            start_points = [best_x.copy()]
            
            # Add a perturbed version to escape potential shallow local minima near the best point
            perturbed = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
            start_points.append(perturbed)
            
            per_start_budget = max(10, polish_budget // len(start_points))
            
            for sp in start_points:
                if call_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                calls_this = [0]
                
                def lbfgs_obj(x):
                    nonlocal calls_this
                    if calls_this[0] >= per_start_budget:
                        return best_f
                    calls_this[0] += 1
                    return eval_func(np.clip(x, -5, 5))
                
                try:
                    res = minimize(
                        lbfgs_obj,
                        sp,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': per_start_budget,
                            'ftol': 1e-12,
                            'gtol': 1e-10,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final Fine-Tuning
        # Use any remaining budget for small random perturbations around the current best
        if best_x is not None and call_count < self.budget:
            remaining = self.budget - call_count
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break
                # Small step size for fine tuning
                y = np.clip(best_x + 0.01 * rng.standard_normal(self.dim), -5, 5)
                eval_func(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
