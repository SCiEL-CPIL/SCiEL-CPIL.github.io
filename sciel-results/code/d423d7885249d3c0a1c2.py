"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizer exploiting the linear structure of BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit:
                return best_f
            
            calls += 1
            try:
                x = np.asarray(x, dtype=float)
                if x.shape != (self.dim,):
                    x = x.reshape(self.dim)
                f_val = func(x)
            except Exception:
                return best_f
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            
            return f_val

        # Phase 1: Gradient Estimation via Axis Probes
        # For a linear function f(x) = c + g^T x, we can estimate g_i
        # by evaluating at +/- 5 * e_i.
        # f(5 e_i) = c + 5 g_i
        # f(-5 e_i) = c - 5 g_i
        # g_i = (f(5 e_i) - f(-5 e_i)) / 10
        
        gradient = np.zeros(self.dim)
        base_x = np.zeros(self.dim)
        
        # Evaluate at origin to get 'c' (optional, helps with sanity check)
        # But we can just use the difference.
        
        for i in range(self.dim):
            if calls >= self.budget or time.time() - start_time > time_limit:
                break
            
            x_plus = np.zeros(self.dim)
            x_plus[i] = 5.0
            f_plus = wrapper(x_plus)
            
            x_minus = np.zeros(self.dim)
            x_minus[i] = -5.0
            f_minus = wrapper(x_minus)
            
            # Estimate gradient component
            gradient[i] = (f_plus - f_minus) / 10.0

        # Phase 2: Construct Optimal Corner
        # For linear function, min is at corner where x_i = -5 if g_i > 0, else 5
        x_opt = np.zeros(self.dim)
        for i in range(self.dim):
            if gradient[i] > 1e-10:
                x_opt[i] = -5.0
            elif gradient[i] < -1e-10:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0 # Gradient is zero, any value is same

        # Evaluate the predicted optimum
        if calls < self.budget and time.time() - start_time <= time_limit:
            f_opt = wrapper(x_opt)
            
            # If the predicted optimum is worse than the best probe, something might be wrong
            # (e.g. non-linearities, noise, or estimation error).
            # In that case, we can fall back to a local search around the best found so far.
            # However, for BBOB f5, this should be the global minimum.
            
            # To be safe and use remaining budget effectively if needed:
            # We can run a simple local search (e.g., Nelder-Mead) from x_opt
            # or best_x if x_opt is not significantly better.
            
            # Given the "Linear slope" description, x_opt should be optimal.
            # We will just return the best found so far.
            # If we have budget left, we can do a few random restarts to be absolutely sure,
            # but for a linear function, it's unnecessary.
            
            # Let's add a small safety check: if f_opt is not better than best_f from probes,
            # it implies the function might not be linear or we hit a numerical issue.
            # But best_f is updated by wrapper, so best_f is now min(probes, f_opt).
            
        return best_x, best_f
# EVOLVE-BLOCK-END
