"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Valley-tracking coordinate descent with adaptive step sizing for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        def time_left():
            return deadline - time.time()
        
        # 1. Global exploration: coarse grid to find the valley
        # The minimizer is in [-4, 4]^5. A 3^5 grid gives 243 points.
        # We use 5 points per dim for a 5^5 = 3125 grid? No, that's too many for 500 budget.
        # Let's stick to a smaller, smarter grid or random sampling.
        # Since Rosenbrock has a long valley, a grid might miss it if it's thin.
        # However, in 5D, the valley is not that thin relative to the box [-5,5].
        # Let's use a 3^5 grid (243 points) to start.
        # Actually, the minimizer is in [-4,4]. Let's use a grid on [-4.5, 4.5] with 3 points: -4.5, 0, 4.5.
        # This is very coarse.
        
        # Better: Random sampling + best point.
        # But the inspiration says "Model the Rosenbrock valley using a coordinate-descent variant".
        # So the key is the local search.
        
        # Let's do a small random search first (50 points) to find a good starting point.
        rng = np.random.default_rng(42)
        best_x = None
        best_f = float('inf')
        
        for _ in range(50):
            if time_left() < 10: break
            x = rng.uniform(-4.5, 4.5, self.dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        # 2. Local Refinement: Adaptive Coordinate Descent (Powell-like but simpler)
        # Rosenbrock's valley is narrow. Standard coordinate descent can be slow if the valley is curved.
        # We will use a "valley-tracking" heuristic:
        # If moving in direction i reduces f significantly, we extend the step in that direction.
        # This is essentially a line search along axes, but with adaptive step sizes.
        
        # We will maintain a step size for each dimension.
        steps = np.ones(self.dim) * 0.5
        
        x = best_x.copy()
        f = best_f
        
        # Coordinate Descent Loop
        # We have ~450 calls left.
        # Each full cycle (5 dims) takes 5 calls (if we just try +/- step).
        # But we can do better.
        
        # Let's use a simpler, robust approach:
        # Powell's method is great for this, but implementing it from scratch is risky.
        # Instead, let's use a "Best-Improvement" Coordinate Descent with line search.
        
        # Actually, the prompt suggests "coordinate-descent variant ... adaptively stretches the search direction along the valley’s curvature".
        # This sounds like we should detect the principal direction of the valley.
        
        # Let's try a different structural change:
        # Use Nelder-Mead but with a very specific initialization and aggressive shrinking/expansion.
        # Or, use L-BFGS-B from scipy? The inspiration mentions L-BFGS-B achieved 4.494 (likely a different metric or task).
        # The task says "Python standard library, NumPy and SciPy are available."
        # Let's use scipy.optimize.minimize with L-BFGS-B or Nelder-Mead.
        # L-BFGS-B is a quasi-Newton method. It requires gradients. We don't have them.
        # But we can use `scipy.optimize.minimize` with `method='Nelder-Mead'` or `method='Powell'`.
        
        # Let's try `scipy.optimize.minimize` with `method='Powell'`.
        # Powell's method is derivative-free and excellent for Rosenbrock.
        
        from scipy.optimize import minimize
        
        # We need to wrap func to count calls and respect time.
        call_count = [0]
        def wrapped_func(x):
            if time_left() < 5:
                return best_f # Return current best to stop optimization
            call_count[0] += 1
            return func(x)
        
        # Powell's method options
        # maxiter is the number of iterations (each iteration involves multiple function calls)
        # We have ~450 calls left. Powell might use ~10-20 calls per iteration in 5D.
        # So ~20-40 iterations.
        
        try:
            res = minimize(wrapped_func, x, method='Powell', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_f:
                best_x = res.x
                best_f = res.fun
        except Exception:
            pass
            
        # If time remains, do a final Nelder-Mead polish
        if time_left() > 10:
            try:
                res2 = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res2.fun < best_f:
                    best_x = res2.x
                    best_f = res2.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
