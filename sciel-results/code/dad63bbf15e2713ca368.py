"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Nelder-Mead with random restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        # Wrapper to count calls and enforce deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            if self.calls > self.budget:
                raise ValueError("Budget exceeded")
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Strategy: 
        # 1. Start at origin (0,0,0,0,0) - often close to shifted minima if shift is small or symmetric
        # 2. Start at a few random points to cover the space
        # 3. Use Nelder-Mead which is efficient for low-dim continuous problems and doesn't need gradients.
        
        # Initial points: Origin + 4 random points
        # We have 500 calls. NM in 5D typically takes ~100-200 calls to converge well.
        # We can afford 2-3 restarts if we are careful, or just 1 robust run.
        # Given the "Bent Cigar" shape (ill-conditioned), NM might struggle if the aspect ratio is huge.
        # However, the problem statement says "Bent cigar" f12. 
        # f12 is typically sum((x_i-1)^2) + 100*sum((x_{i+1}-1)^2) ... no, that's Rosenbrock.
        # BBOB f12 is "Bent Cigar". It has a narrow valley.
        # NM is not great for narrow valleys. 
        # But the seed is (1+1)-ES which is also not great.
        # Let's try a more robust approach: CMA-ES is not in stdlib/scipy directly as a simple function (it's in pygmo or cma package, not guaranteed).
        # SciPy has `differential_evolution` (global) and `minimize` (local).
        # `differential_evolution` is stochastic and might use too many calls.
        # Let's stick to `minimize` with Nelder-Mead but with a better initialization.
        
        # Actually, for ill-conditioned problems like Bent Cigar, Nelder-Mead can fail if the simplex degenerates.
        # A better local method for this might be L-BFGS-B if we could estimate gradients, but we can't.
        # Powell's method is also in SciPy and is derivative-free, often better for ill-conditioned problems than NM.
        
        # Let's use Powell's method.
        
        start_points = [np.zeros(self.dim)]
        # Add a few random starts to be safe against shifts
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, self.dim))
        
        # We have 500 calls.
        # Powell's method in 5D: each cycle evaluates dim+1 points roughly.
        # 500 / 6 ~ 80 cycles. Should be enough for 1-2 restarts.
        
        # We will run Powell on the best starting point first? 
        # Or just run it on origin, then if time/calls remain, on a random point.
        
        # Let's try running Powell on origin first.
        # If it converges quickly, we might have budget left for a restart.
        
        # However, `minimize` with Powell might not stop early easily without maxiter.
        # We can set maxiter to a value that fits the budget.
        
        # Let's try a different tactic: 
        # Use `differential_evolution`? No, too many calls.
        # Use `basinhopping`? It uses a local optimizer (L-BFGS-B or NM) and a global kick.
        # `basinhopping` is in SciPy. It is designed for global optimization of local minima.
        # It uses a local optimizer (default L-BFGS-B) which requires gradients? No, L-BFGS-B approximates them.
        # But L-BFGS-B on a black box might be slow or inaccurate with finite differences.
        # `basinhopping` with `minimizer_kwargs={'method': 'Nelder-Mead'}` is safer.
        
        # Let's try `basinhopping`.
        # It will do a series of kicks and local optimizations.
        # We need to control the number of iterations to stay within budget.
        
        # Alternative: Just use Powell's method directly.
        
        # Let's implement a simple restart strategy with Powell.
        
        for start in start_points:
            if time.time() > self.deadline or self.calls > self.budget - 10:
                break
            
            try:
                # Powell's method
                # maxiter: max number of iterations. In 5D, 1 iteration ~ 5-10 calls.
                # Let's allocate ~100 calls per restart.
                # 100 calls / 5D ~ 20 iterations.
                res = minimize(wrapped_func, start, method='Powell', 
                               options={'maxiter': 20, 'maxfev': 100})
            except (TimeoutError, ValueError):
                break
            except Exception:
                # Catch other exceptions to keep going if possible? 
                # The contract says an exception later keeps the calls made.
                # But we want to return the best result.
                pass

        if self.best_x is None:
             # Fallback if no calls were made or something went wrong
             self.best_x = np.zeros(self.dim)
             self.best_f = func(self.best_x)
             self.calls = 1

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
