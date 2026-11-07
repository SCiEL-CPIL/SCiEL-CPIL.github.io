"""Combined strategy: Fast affine model identification with robust validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizes BBOB f5 (Linear Slope).
    Strategy:
    1. Rapidly identify the linear gradient using forward differences at x=0.
    2. Compute the analytic optimum (a vertex of the box).
    3. Validate the linearity assumption with a few random points.
    4. If valid, fill the remaining budget with evaluations at the optimum to maximize AOCC.
    5. If invalid, fall back to a robust local search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, leave 10s margin
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Estimate gradient using forward differences at x=0.
        # We need d+1 calls: f(0) and f(e_i) for i in 1..d.
        # This is faster than central differences (2d+1 calls) and sufficient for linear functions.
        
        x0 = np.zeros(dim)
        if time.time() > deadline or calls >= budget:
            return best_x, best_fx

        f0 = func(x0)
        calls += 1
        update_best(x0, f0)
        
        # Store f(e_i) to compute gradient components
        f_e = np.zeros(dim)
        grad = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls >= budget:
                break
            xi = x0.copy()
            xi[i] = 1.0
            fi = func(xi)
            calls += 1
            update_best(xi, fi)
            f_e[i] = fi
            grad[i] = fi - f0 # Slope along axis i

        # Phase 2: Compute analytic optimum for the linear model
        # Minimize f(x) = f0 + grad^T x subject to -5 <= x_i <= 5
        x_opt = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_opt[i] = -5.0
            elif grad[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the predicted optimum
        if time.time() > deadline or calls >= budget:
            return best_x, best_fx
            
        f_opt = func(x_opt)
        calls += 1
        update_best(x_opt, f_opt)

        # Phase 3: Validate linearity
        # If the function is truly linear, f(x) should match f0 + grad^T x for any x.
        # We check 3 random points. If the model holds, we assume we found the global min.
        
        rng = np.random.default_rng(42)
        is_linear = True
        check_count = 3
        
        for _ in range(check_count):
            if time.time() > deadline or calls >= budget:
                break
            x_check = rng.uniform(-5, 5, dim)
            f_check = func(x_check)
            calls += 1
            update_best(x_check, f_check)
            
            # Predicted value
            f_pred = f0 + np.dot(grad, x_check)
            
            # Check residual
            # Tolerance: The function values are in [-100, 100]. 
            # A deviation > 1e-3 suggests non-linearity or noise.
            if abs(f_check - f_pred) > 1e-3:
                is_linear = False
                break

        if is_linear:
            # The function is linear. x_opt is the global minimum.
            # To maximize AOCC, we should keep the "best value" as low as possible for as long as possible.
            # Since we already found the global min, we can just call func(x_opt) repeatedly.
            # This ensures b_i stays at f_opt for all remaining i.
            while calls < budget and time.time() < deadline:
                f_val = func(x_opt)
                calls += 1
                # No need to update best if it's the same, but good for safety
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = x_opt.copy()
        else:
            # Fallback: The function is not linear.
            # Use remaining budget for a simple local search or random restarts.
            # Given the task is f5 (Linear Slope), this case should rarely trigger.
            # We'll do a few random walks around the current best.
            while calls < budget and time.time() < deadline:
                # Simple random perturbation
                step = 0.5
                x_new = best_x + rng.normal(0, step, dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                update_best(x_new, f_new)
                
                # If we find something better, we might refine, but for now keep it simple.

        return best_x, best_fx

# EVOLVE-BLOCK-END
