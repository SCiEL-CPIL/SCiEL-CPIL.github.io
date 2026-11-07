"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer using a coarse global scout followed by Powell's method."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        call_count = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        # Exception to signal budget or time exceeded
        class SearchStop(Exception):
            pass

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            
            # Time check
            if time.time() > deadline:
                raise SearchStop("Time limit exceeded")
            
            # Budget check
            call_count += 1
            if call_count > self.budget:
                raise SearchStop("Budget exceeded")
            
            # Ensure x is a valid array of correct shape
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # 1. Global Scout Phase
        # We use a grid of 3 points per dimension: {-3, 0, 3}
        # This gives 3^5 = 243 points.
        # This is a significant increase over the previous 5 points, but leaves 257 for local search.
        # The range [-3, 3] is chosen because the minimum is guaranteed to be in [-4, 4],
        # and the Rosenbrock valley typically doesn't extend too far outside this for the standard function.
        # Even with shifts, this grid should catch the basin of attraction.
        
        scout_values = [-3.0, 0.0, 3.0]
        scout_points = []
        # Generate all combinations
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            scout_points.append(np.array([scout_values[i], scout_values[j], scout_values[k], scout_values[l], scout_values[m]]))
        
        # Sort scout points? No, just evaluate them.
        # We can prioritize center? 
        # Actually, let's just evaluate them all if budget allows.
        # 243 points is fine.
        
        for x in scout_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            try:
                wrapped_func(x)
            except SearchStop:
                break

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            return np.ones(self.dim), np.inf

        # 2. Optimization Phase: Use Powell's method
        # Start from the best point found in the scout phase.
        # Powell is robust for Rosenbrock.
        
        try:
            minimize(
                wrapped_func, 
                best_x, 
                method='Powell',
                options={
                    'xtol': 1e-10, 
                    'ftol': 1e-12, 
                    'maxiter': 1000, 
                    'maxfev': self.budget - call_count
                }
            )
        except SearchStop:
            pass
        except Exception:
            # Catch any other unexpected errors to ensure we return a valid result
            pass

        # 3. If we still have budget and time, try a second local run from the second best scout point?
        # This might help if the first local run got stuck in a local minimum (less likely for Rosenbrock, but possible with shifts).
        # However, tracking the second best is complex with the current wrapped_func.
        # Let's stick to one strong run to ensure we finish safely, as Powell is quite effective.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
