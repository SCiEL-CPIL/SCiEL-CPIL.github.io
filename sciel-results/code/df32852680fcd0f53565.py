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
        
        # Wrapped function to count calls
        original_func = func
        
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f if np.isfinite(best_f) else 0.0
            if not time_ok():
                return best_f if np.isfinite(best_f) else 0.0
            
            x = np.clip(x, -5, 5)
            f_val = float(original_func(x))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Gradient function using central differences
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

        # Phase 1: Coarse global search using CMA-ES
        # CMA-ES is derivative-free and robust to the narrow valley of Discus.
        # We allocate ~100 calls to this phase.
        cma_budget = 100
        cma_calls = 0
        
        # Initialize CMA-ES manually to control budget strictly
        # Standard CMA-ES implementation is complex to wrap perfectly with budget checks
        # without a library like cma. Let's use a simple evolutionary strategy or
        # just rely on multi-start L-BFGS-B but with better starts.
        
        # Actually, implementing CMA-ES from scratch is error-prone.
        # Let's stick to the "Inspiration 1" strategy which scored well:
        # Multi-start L-BFGS-B with diverse initial points and strict budget management.
        # But we will improve the start points and the allocation.
        
        # Generate diverse start points
        rng = np.random.default_rng(42)
        starts = []
        
        # 1. Center
        starts.append(np.zeros(d))
        
        # 2. Axis-aligned points (exploiting structure if any, though Discus is rotated)
        for i in range(d):
            pt_pos = np.zeros(d)
            pt_neg = np.zeros(d)
            pt_pos[i] = 4.0
            pt_neg[i] = -4.0
            starts.append(pt_pos)
            starts.append(pt_neg)
            
        # 3. Random points
        n_random = 10
        for _ in range(n_random):
            starts.append(rng.uniform(-4.5, 4.5, d))
            
        # 4. Add current best (if any)
        if best_f < float('inf'):
            starts.insert(0, best_x.copy())
            
        # Shuffle to randomize order of exploration
        rng.shuffle(starts)
        
        # Phase 2: Multi-start L-BFGS-B
        # We will run L-BFGS-B from each start point.
        # To ensure early progress, we run them sequentially but with a strict call limit per run.
        # If a run converges quickly, we move to the next.
        
        # Estimate calls per L-BFGS-B iteration: 1 (func) + 2*d (grad) = 11 for d=5.
        # We have ~400 calls left.
        # Let's allow up to 20 iterations per start (220 calls).
        # This allows for ~1-2 full convergences or several partial ones.
        
        for start_pt in starts:
            if calls >= budget or not time_ok():
                break
            
            # Ensure start point is within bounds
            start_pt = np.clip(start_pt, -5, 5)
            
            # Evaluate start point
            f_start = wrapped_func(start_pt)
            
            if calls >= budget or not time_ok():
                break
            
            # Run L-BFGS-B
            try:
                res = minimize(
                    wrapped_func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=grad_func, 
                    bounds=[(-5, 5)] * d,
                    options={
                        'maxiter': 20, 
                        'maxls': 30, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8,
                        'maxfun': max(5, budget - calls)
                    }
                )
                
                # Update best
                # Note: minimize might not update best_x/best_f if it didn't call wrapped_func
                # in a way that triggers the update (it does, via jac and func).
                # But just in case, check res.fun
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
                    
            except Exception:
                pass

            # If we have significant budget left and found a good point, do a polish
            # This is a refinement step from Inspiration 2
            if calls < budget - 50 and time_ok():
                try:
                    res2 = minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        jac=grad_func, 
                        bounds=[(-5, 5)] * d,
                        options={
                            'maxiter': 10, 
                            'maxls': 20, 
                            'ftol': 1e-15, 
                            'gtol': 1e-10,
                            'maxfun': max(5, budget - calls)
                        }
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
