# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer using L-BFGS-B with random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        
        # We will use a portion of the budget for local optimization
        # and keep some for exploration/restarts if needed.
        # L-BFGS-B is very efficient.
        
        # Strategy:
        # 1. Try a few random starting points.
        # 2. For each, run L-BFGS-B with a limited number of function evaluations.
        # 3. Keep the best result.
        
        # Let's allocate budget dynamically.
        # L-BFGS-B typically converges quickly for smooth-ish functions.
        # f13 (Sharp Ridge) is non-smooth (has a ridge), but L-BFGS-B can still make progress.
        # However, it might get stuck. Random restarts help.
        
        # Let's try 5 restarts.
        # Each restart gets budget / 5 evaluations roughly?
        # But we need to be careful not to exceed total budget.
        
        # Actually, `minimize` with method 'L-BFGS-B' has `maxiter` and `maxfun`.
        # We should track calls globally.
        
        calls_made = 0
        
        def wrapper(x):
            nonlocal calls_made
            if calls_made >= budget:
                return func(x)  # Should not happen if we control maxfun, but safe
            calls_made += 1
            return func(x)
        
        # We need to ensure we don't exceed budget.
        # Let's divide the budget among a few restarts.
        num_restarts = 5
        evals_per_restart = budget // num_restarts
        
        # Initial random points
        rng = np.random.default_rng(42)
        
        for i in range(num_restarts):
            if calls_made >= budget:
                break
            
            # Choose a starting point
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            # Run L-BFGS-B
            # maxfun is the maximum number of function evaluations.
            # We have remaining budget: budget - calls_made
            # But we want to spread it.
            # Let's just cap maxfun to the remaining budget to be safe, 
            # but ideally we want to stop early if it converges.
            
            # L-BFGS-B parameters
            # ftol: relative error tolerance on the order of the function value
            # gtol: absolute error tolerance on the order of the gradient norm
            
            try:
                res = minimize(
                    wrapper,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 50,
                        'maxfun': max(10, budget - calls_made), # Cap at remaining budget
                        'ftol': 1e-10,
                        'gtol': 1e-5,
                        'disp': False
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If optimization fails, try to get the current best from res if possible
                # or just ignore and try next restart
                pass
            
            # If we've used most of the budget, break
            if calls_made > budget * 0.9:
                break
        
        # If no valid best_x was found (shouldn't happen), fallback
        if np.isnan(best_x).any():
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
