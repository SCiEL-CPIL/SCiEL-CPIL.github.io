"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead optimizer for the BBOB Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        timeout = 295.0  # Safety margin before 330s kill

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def call_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse search to find the basin of attraction
        # Use a large initial simplex to cover the domain [-5, 5]^5
        # We will use Nelder-Mead from scipy with a large initial spread
        
        # Initial guess: center of the box
        x0 = np.zeros(dim)
        
        # Create initial simplex manually to ensure we stay within bounds and cover space
        # Standard Nelder-Mead starts with x0 and x0 + delta * e_i
        # We want a large delta to cover the space quickly
        delta = 4.0  # Covers most of the [-5,5] range from center
        
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            
        # Evaluate initial simplex
        fvals = []
        for i in range(dim + 1):
            if call_count >= budget or time.time() - start_time > timeout:
                break
            fvals.append(call_func(simplex[i]))
            
        if len(fvals) < dim + 1:
            return best_x, best_fx

        # Run Nelder-Mead
        # maxiter is not strictly budget-limited in scipy, so we wrap it
        # We need to stop when call_count exceeds budget or time runs out
        
        # We'll run NM in a loop, checking budget between iterations isn't possible directly
        # So we set maxiter high and rely on the time/budget check inside the objective?
        # No, scipy's minimize doesn't support early stopping via callback easily without
        # raising an exception.
        
        # Better approach: Use scipy's minimize with a wrapper that raises an exception
        # when budget/time is exceeded, or just run it and hope it finishes.
        # Given dim=5 and 500 budget, NM should converge or hit maxiter quickly.
        # Let's set maxiter to a safe number that won't exceed budget too much.
        # Each NM iteration uses ~dim+1 = 6 calls on average, but can use more.
        # 500 budget / 6 calls/iter ~ 80 iterations.
        
        # However, NM can get stuck.
        # Let's try running it. If it uses too many calls, we just return best found.
        
        # We need a way to stop scipy.
        # We can pass a custom objective that checks budget.
        
        def objective(x):
            if call_count >= budget or time.time() - start_time > timeout:
                raise Exception("Budget or time exceeded")
            return call_func(x)
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000,  # High, will be stopped by exception
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            pass
        except:
            pass

        # If we still have budget and time left, we might do a second polish
        # or just return.
        # The "Inspiration" mentions Nelder-Mead polish.
        # Since we already used NM, we can just return.
        
        # Ensure we didn't exceed budget in the last few calls
        return best_x, best_fx

# EVOLVE-BLOCK-END
