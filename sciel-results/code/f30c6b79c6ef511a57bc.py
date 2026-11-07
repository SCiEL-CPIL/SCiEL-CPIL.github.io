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
        deadline = time.time() + 285
        rng = np.random.default_rng(42)
        
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        # Phase 1: Sparse coordinate-descent with adaptive step sizes
        # Rapidly traverse the search space to find a promising basin
        explore_budget = min(100, budget // 3)
        
        # Start with a few random points to get initial diversity
        for i in range(min(10, explore_budget)):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            calls += 1
            if fx < fx_best:
                x_best = x.copy()
                fx_best = fx
        
        # Coordinate descent with adaptive step sizes
        step_sizes = np.ones(dim) * 2.0
        for i in range(explore_budget - calls):
            if calls >= budget or time.time() > deadline:
                break
            
            # Try each coordinate with current step size
            improved = False
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive direction
                x_try = x_best.copy()
                x_try[d] += step_sizes[d]
                x_try = np.clip(x_try, -5, 5)
                fx_try = func(x_try)
                calls += 1
                
                if fx_try < fx_best:
                    x_best = x_try.copy()
                    fx_best = fx_try
                    improved = True
                    # Increase step size if improvement
                    step_sizes[d] *= 1.2
                else:
                    # Try negative direction
                    x_try = x_best.copy()
                    x_try[d] -= step_sizes[d]
                    x_try = np.clip(x_try, -5, 5)
                    fx_try = func(x_try)
                    calls += 1
                    
                    if fx_try < fx_best:
                        x_best = x_try.copy()
                        fx_best = fx_try
                        improved = True
                        step_sizes[d] *= 1.2
                    else:
                        # Decrease step size if no improvement
                        step_sizes[d] *= 0.7
            
            # If no improvement in any coordinate, reduce all step sizes
            if not improved:
                step_sizes *= 0.5
                if np.all(step_sizes < 0.01):
                    break
        
        # Phase 2: Nelder-Mead local refinement
        # Exploit the smooth curvature of the Rosenbrock valley
        remaining_budget = budget - calls
        if remaining_budget > 20:
            try:
                # Initialize simplex around the best point
                x0 = x_best.copy()
                simplex = [x0]
                for d in range(dim):
                    x_simplex = x0.copy()
                    # Small perturbation in each dimension
                    x_simplex[d] += 0.1 * step_sizes[d] if step_sizes[d] > 0.01 else 0.1
                    x_simplex = np.clip(x_simplex, -5, 5)
                    simplex.append(x_simplex)
                
                result = minimize(
                    func, x0, method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining_budget, 200),
                        'xatol': 1e-6,
                        'fatol': 1e-8
                    }
                )
                
                if result.fun < fx_best:
                    x_best = result.x.copy()
                    fx_best = result.fun
                calls += result.nfev
            except Exception:
                pass
            
            # If we still have budget, try a few more local optimizations
            remaining_budget = budget - calls
            if remaining_budget > 20:
                for restart in range(min(3, remaining_budget // 30)):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Generate a point near the current best
                    perturbation = 0.3 * rng.standard_normal(dim)
                    x_start = np.clip(x_best + perturbation, -5, 5)
                    fx_start = func(x_start)
                    calls += 1
                    
                    if fx_start < fx_best:
                        x_best = x_start.copy()
                        fx_best = fx_start
                    
                    try:
                        result = minimize(
                            func, x_start, method='Nelder-Mead',
                            options={
                                'maxiter': min(remaining_budget // 2, 100),
                                'xatol': 1e-6,
                                'fatol': 1e-8
                            }
                        )
                        if result.fun < fx_best:
                            x_best = result.x.copy()
                            fx_best = result.fun
                        calls += result.nfev
                    except Exception:
                        pass
        
        return x_best, fx_best
