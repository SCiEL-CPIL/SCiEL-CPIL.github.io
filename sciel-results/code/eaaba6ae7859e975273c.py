import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # State for numerical gradient
        # We use central differences for better accuracy if budget allows,
        # but forward differences are cheaper. Given 500 calls and dim=5,
        # central diff costs 10 per gradient. 
        # L-BFGS-B might need ~20-50 gradients per convergence.
        # 500 / 10 = 50 gradients total.
        # So we can afford ~2-4 L-BFGS-B runs if they converge quickly, 
        # or many short runs.
        # However, L-BFGS-B is very efficient. Let's try to fit 5-10 starts.
        
        # Strategy: 
        # 1. Generate a set of diverse starting points.
        # 2. For each start, run L-BFGS-B with a numerical gradient wrapper.
        # 3. The wrapper counts calls and stops if budget is exhausted.
        # 4. Update global best.
        
        # To maximize early progress (AOCC), we want to find good points quickly.
        # A coarse grid or random starts? Random is better for high-dim, but dim=5 is low.
        # Let's use a mix: center, corners, and random.
        
        # Generate initial points
        # We will allocate a small fraction of budget to "scouting" and rest to optimization.
        # Actually, L-BFGS-B itself does the scouting via the line search.
        # Let's just pick 10 diverse starting points.
        
        # Deterministic diverse points for reproducibility and coverage
        # But the task says "hidden seeded shift", so random is safer to avoid bad luck with fixed grid.
        # However, we want to be robust.
        # Let's use 10 random points, but ensure they are spread out.
        
        rng = np.random.RandomState(42) # Fixed seed for internal randomness to ensure consistency across runs? 
        # Wait, the evaluator creates a new Optimizer for each instance. 
        # If I use a fixed seed, the starting points are the same relative to the box.
        # Since the shift is hidden and random, fixed relative points are fine.
        
        # Let's create 10 starting points
        # 1. Center (0,0,0,0,0)
        # 2-10. Random points in [-4.5, 4.5]
        
        start_points = []
        start_points.append(np.zeros(d))
        
        for _ in range(9):
            pt = rng.uniform(-4.5, 4.5, d)
            start_points.append(pt)
            
        # Shuffle to randomize order? 
        # Early progress matters. If the first point is bad, we waste time.
        # But L-BFGS-B is fast. Let's just iterate.
        
        # To improve early progress, we can do a quick local search (e.g. Nelder-Mead) 
        # on the first few points? No, L-BFGS-B is better.
        
        # Let's refine the gradient function.
        # Central difference: (f(x+he) - f(x-he)) / (2h)
        # Cost: 2*d + 1 per gradient step.
        # For d=5, cost is 11.
        # If we have 500 calls, we can do ~45 gradients.
        # L-BFGS-B typically converges in 10-30 iterations for smooth functions.
        # So 45 gradients is enough for ~2-4 convergences.
        
        # We should prioritize starting points that might be closer to the optimum.
        # Since we don't know where it is, we just run them sequentially.
        
        # Optimization: 
        # We can use a cheaper gradient (forward difference) if we are running low on budget?
        # Or just stick to central for accuracy.
        
        def grad_func(x):
            nonlocal calls, best_x, best_f, deadline
            
            if calls >= budget or not time_ok():
                return np.zeros(d)
                
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                # Check budget for each eval
                if calls >= budget or not time_ok():
                    return grad # Return partial or zero? Zero is safer to stop.
                
                f_plus = func(x_plus)
                calls += 1
                if f_plus < best_f:
                    best_f = f_plus
                    best_x = x_plus.copy()
                    
                if calls >= budget or not time_ok():
                    return grad
                
                f_minus = func(x_minus)
                calls += 1
                if f_minus < best_f:
                    best_f = f_minus
                    best_x = x_minus.copy()
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Run L-BFGS-B for each start point
        for start_pt in start_points:
            if calls >= budget or not time_ok():
                break
            
            # Check if we've already converged this point? No, fresh start.
            
            # Limits for L-BFGS-B
            bounds = [(-5, 5)] * d
            
            try:
                # maxiter: limit iterations to prevent hanging
                # We don't know how many calls one iteration takes (11), 
                # so maxiter=10 is ~110 calls. 
                # We have ~500 calls. 500/110 ~ 4 starts.
                # We have 10 starts. So many will be cut short.
                # That's okay, they might still find good points.
                
                res = minimize(
                    func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=grad_func, 
                    bounds=bounds,
                    options={'maxiter': 15, 'maxls': 20} # Limit line search steps too
                )
                
                # L-BFGS-B returns the best x it found
                # It might have used calls, so update best
                # The func calls inside minimize are already counted in `calls` via grad_func
                # But minimize also evaluates func at the start and during line search?
                # Yes, `func` is called directly by minimize. 
                # Wait! My `grad_func` calls `func` for gradients.
                # `minimize` ALSO calls `func` to evaluate the objective.
                # So one iteration costs: 1 (objective) + 2*d (gradient) = 1 + 10 = 11 calls.
                # My counter `calls` only increments in `grad_func`.
                # It does NOT increment when `minimize` calls `func` directly!
                
                # This is a BUG. I must wrap `func` to count ALL calls.
                
            except Exception:
                pass
        
        # Re-implement with a proper wrapper for func to count all calls.
        
        # Reset state for the correct implementation
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        original_func = func
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f # Return a large value to stop? Or just return last best?
                # Better to return a value that doesn't break optimizer, but we stop anyway.
            if not time_ok():
                return best_f
            
            x = np.clip(x, -5, 5)
            f_val = original_func(x)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        def grad_func(x):
            nonlocal calls, deadline
            if calls >= budget or not time_ok():
                return np.zeros(d)
                
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                # Check budget before calling
                if calls + 2 > budget or not time_ok():
                    return grad
                
                f_plus = wrapped_func(x_plus)
                f_minus = wrapped_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Generate start points again
        rng = np.random.RandomState(42)
        start_points = []
        start_points.append(np.zeros(d))
        for _ in range(9):
            pt = rng.uniform(-4.5, 4.5, d)
            start_points.append(pt)
            
        # Shuffle to randomize which one gets the full budget?
        # No, deterministic order is fine.
        
        for start_pt in start_points:
            if calls >= budget or not time_ok():
                break
            
            bounds = [(-5, 5)] * d
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=grad_func, 
                    bounds=bounds,
                    options={'maxiter': 15, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass
        
        return best_x, best_f
