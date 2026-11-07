"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive local restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Wrap the objective to count calls and enforce budget
        call_count = [0]
        max_calls = budget

        def wrapped(x):
            if call_count[0] >= max_calls:
                return float('inf')
            x = np.asarray(x, dtype=np.float64)
            # Ensure finite
            x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            if not np.isfinite(val):
                return 1e30
            return float(val)

        # Strategy: Use a mix of global exploration and local exploitation.
        # Since the function is Rosenbrock rotated, L-BFGS-B is very effective
        # once in the basin. We will do several random restarts with L-BFGS-B.
        
        # Reserve some budget for random sampling if needed, but primarily use
        # the optimizer which consumes budget internally via gradient steps.
        # Note: scipy.optimize.minimize with L-BFGS-B uses multiple function evaluations
        # per iteration. We need to be careful not to exceed the budget.
        
        # A safer approach for budget control:
        # Implement a custom loop that does:
        # 1. Pick a starting point.
        # 2. Run a few steps of a simple gradient-free or gradient-based method
        #    with strict budget checking.
        # However, the task says "Black-box continuous optimizer". We don't have gradients.
        # So we should use a gradient-free method or approximate gradients.
        # L-BFGS-B in scipy uses finite differences for gradients if not provided,
        # which is expensive (2*dim+1 calls per gradient).
        
        # Let's use a different strategy: 
        # 1. Global search phase: Random restarts with Nelder-Mead or Powell?
        # Nelder-Mead is robust and doesn't need gradients.
        # Powell is also good.
        
        # Given the 500 budget and dim=5, Nelder-Mead is a solid choice.
        # We can do multiple restarts.
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Random sampling to find a good initial point?
        # Or just start Nelder-Mead from several random points.
        
        # Let's allocate budget.
        # Each Nelder-Mead iteration uses ~dim+1 = 6 calls.
        # 500 / 6 ~ 83 iterations per run.
        # We can do maybe 5-10 restarts if we stop early when convergence is reached.
        
        # Strategy:
        # - Start with a random point.
        # - Run Nelder-Mead with maxiter set such that we don't exceed budget.
        # - If we improve, update best.
        # - Repeat with new random start if budget allows.
        
        # To be safe, we'll implement a loop that checks budget before each step.
        # But scipy.optimize.minimize doesn't allow fine-grained budget control mid-run easily.
        # So we estimate the cost.
        
        # Alternative: Use a simpler, more controllable algorithm like CMA-ES?
        # CMA-ES is available in scipy? No, not in standard scipy. 
        # But we can implement a simple version or use another method.
        
        # Let's stick to scipy.optimize.minimize with method='Nelder-Mead'.
        # We will set maxiter to a value that keeps us within budget.
        # Each iteration ~ dim+1 calls.
        # Let's try 5 restarts.
        # Budget per restart: ~ 500 / 5 = 100 calls.
        # maxiter ~ 100 / 6 = 16 iterations. That's quite low.
        
        # Maybe fewer restarts? 2 restarts?
        # 500 / 2 = 250 calls per restart.
        # maxiter ~ 250 / 6 = 41 iterations.
        
        # Let's try a dynamic approach:
        # Start with one big run. If it converges, stop.
        # If not, and we have budget left, restart.
        
        # But how do we know if it converged?
        # We can check the improvement.
        
        # Let's try a different tactic: 
        # Use Powell's method. It is also derivative-free and often efficient.
        
        # Actually, let's use a hybrid:
        # 1. Do a few random points to find a low-f region.
        # 2. Then use Nelder-Mead from the best point found.
        
        # Phase 1: Random Sampling
        n_random = 20
        for i in range(n_random):
            if call_count[0] >= max_calls:
                break
            x = rng.uniform(-5, 5, dim)
            f = wrapped(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local Optimization from best point
        if call_count[0] < max_calls:
            # Estimate remaining budget
            remaining = max_calls - call_count[0]
            # Nelder-Mead uses ~dim+1 calls per iteration
            # Let's allow up to remaining // (dim+1) iterations
            max_iter = max(1, remaining // (dim + 1))
            
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # Phase 3: If we still have budget, try another restart
        if call_count[0] < max_calls and call_count[0] < max_calls - 50:
            # Start from a new random point, but closer to the best?
            # Or just another random point.
            x_start = rng.uniform(-5, 5, dim)
            remaining = max_calls - call_count[0]
            max_iter = max(1, remaining // (dim + 1))
            
            try:
                res = minimize(
                    wrapped,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
