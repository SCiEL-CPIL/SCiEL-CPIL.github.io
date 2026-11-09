import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        
        # Phase 1: Linear Gradient Estimation
        # We use h=5 to hit the box boundaries directly
        h = 5.0
        grad = np.zeros(self.dim)
        calls_used = 1
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            xi = np.zeros(self.dim)
            xi[i] = h
            fi = func(xi)
            calls_used += 1
            grad[i] = (fi - f0) / h
            
            # Update best if this probe is better
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
                
        # Construct optimal vertex based on gradient
        x_star = np.zeros(self.dim)
        for i in range(self.dim):
            if grad[i] > 0:
                x_star[i] = -h
            else:
                x_star[i] = h
        
        # Evaluate the predicted optimal vertex
        if time.time() < deadline:
            f_star = func(x_star)
            calls_used += 1
            if f_star < best_f:
                best_f = f_star
                best_x = x_star.copy()
        
        # Check linearity consistency
        # Predicted f at x_star: f0 + grad @ x_star
        f_predicted = f0 + np.dot(grad, x_star)
        
        # If the prediction is off, the function might not be linear (or noise)
        # In that case, use remaining budget for local optimization
        if abs(f_star - f_predicted) > 1e-5:
            # Use Nelder-Mead for the remaining budget
            # We have budget - calls_used calls left
            remaining = self.budget - calls_used
            if remaining > 10 and time.time() < deadline:
                # Wrap func to count calls and enforce deadline
                call_count = [0]
                def wrapped_func(x):
                    call_count[0] += 1
                    if time.time() > deadline:
                        return best_f # Return current best to stop optimization
                    return func(x)
                
                try:
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
                    
        return best_x, best_f
