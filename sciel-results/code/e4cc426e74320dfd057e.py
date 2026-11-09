import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Evaluate Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        update_best(x0, f0)

        # 2. Sequential Coordinate Descent to find the optimal vertex
        # For a linear function, the minimum is at a vertex.
        # Evaluate each axis at +5 and -5 from center to determine gradient sign.
        # Greedily fix coordinates with largest observed improvement.
        grad_sign = np.zeros(dim)  # +1 means decrease when going to -5, -1 means decrease when going to +5
        improvements = np.zeros(dim)  # magnitude of improvement for each axis
        
        for i in range(dim):
            if calls_made >= budget or time.time() > deadline:
                break
            xi_pos = np.zeros(dim)
            xi_pos[i] = 5.0
            fi_pos = func(xi_pos)
            calls_made += 1
            update_best(xi_pos, fi_pos)
            
            if calls_made >= budget or time.time() > deadline:
                # Only got one evaluation; assume gradient direction based on fi_pos vs f0
                if fi_pos < f0:
                    grad_sign[i] = -1  # going to +5 decreased, so gradient is negative
                    improvements[i] = f0 - fi_pos
                else:
                    grad_sign[i] = 1   # going to +5 increased, so gradient is positive
                    improvements[i] = fi_pos - f0
                continue
                
            xi_neg = np.zeros(dim)
            xi_neg[i] = -5.0
            fi_neg = func(xi_neg)
            calls_made += 1
            update_best(xi_neg, fi_neg)
            
            # Determine which direction gives better value
            if fi_pos < fi_neg:
                # +5 is better, so gradient is negative (function decreases as x_i increases)
                grad_sign[i] = -1
                improvements[i] = f0 - fi_pos
            else:
                # -5 is better, so gradient is positive (function decreases as x_i decreases)
                grad_sign[i] = 1
                improvements[i] = f0 - fi_neg

        # 3. Predict Optimal Vertex
        # grad_sign[i] = 1 means gradient is positive, so min at x_i = -5
        # grad_sign[i] = -1 means gradient is negative, so min at x_i = +5
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad_sign[i] > 0:
                x_pred[i] = -5.0
            else:
                x_pred[i] = 5.0
        
        # Also try the opposite vertex to be safe
        x_pred_alt = np.zeros(dim)
        for i in range(dim):
            if grad_sign[i] > 0:
                x_pred_alt[i] = 5.0
            else:
                x_pred_alt[i] = -5.0
        
        if calls_made < budget and time.time() < deadline:
            f_pred = func(x_pred)
            calls_made += 1
            update_best(x_pred, f_pred)
        
        # Evaluate the alternative vertex if we have budget
        if calls_made < budget and time.time() < deadline:
            f_pred_alt = func(x_pred_alt)
            calls_made += 1
            update_best(x_pred_alt, f_pred_alt)
            
        # 4. Verification/Refinement
        # For a truly linear function, the vertex is optimal.
        # Check if best is at a vertex (all coordinates are ±5)
        is_vertex = np.all(np.abs(np.abs(best_x) - 5.0) < 1e-6)
        
        if is_vertex:
            # We found a vertex. For linear functions, this should be optimal.
            # Use remaining budget for lightweight verification with small perturbations.
            x = best_x
            fx = best_f
            sigma = 0.5
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 0.8
                else:
                    sigma *= 0.85
                    
                if sigma < 1e-8:
                    break
        else:
            # Not at a vertex - do a more thorough local search
            x = best_x
            fx = best_f
            sigma = 2.0
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.02, 5.0)
                else:
                    sigma *= 0.95
                    
                if sigma < 1e-8:
                    break

        return best_x, best_f
