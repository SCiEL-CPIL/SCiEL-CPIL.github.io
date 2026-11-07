import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.deadline = time.time() + 290  # Safety margin for 300s limit

    def __call__(self, func):
        best_x = np.zeros(self.dim)
        best_f = np.inf
        x_center = np.zeros(self.dim)
        
        # 1. Evaluate center
        f_center = func(x_center)
        self.calls += 1
        best_f = f_center
        best_x = x_center.copy()
        
        if self.calls >= self.budget or time.time() > self.deadline:
            return best_x, best_f

        # 2. Determine signs for each dimension
        # Probe with a step size. 1.0 is safe within [-5, 5]
        probe_step = 1.0
        signs = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            x_pos = x_center.copy()
            x_pos[i] += probe_step
            f_pos = func(x_pos)
            self.calls += 1
            
            if self.calls >= self.budget or time.time() > self.deadline:
                break
                
            x_neg = x_center.copy()
            x_neg[i] -= probe_step
            f_neg = func(x_neg)
            self.calls += 1
            
            # If f_pos < f_center, slope is negative -> min at +5
            # If f_neg < f_center, slope is positive -> min at -5
            # f(x) = a_i * x_i. 
            # f(center + d) - f(center) = a_i * d.
            # If a_i * d > 0 (f_pos > f_center), a_i > 0 (since d>0). Min at -5.
            # If a_i * d < 0 (f_pos < f_center), a_i < 0. Min at +5.
            
            if f_pos < f_center:
                signs[i] = 1.0  # Go to +5
            elif f_neg < f_center:
                signs[i] = -1.0 # Go to -5
            else:
                # Flat or noise? Assume 0 or keep current best direction
                signs[i] = 0.0 

        # 3. Construct candidate vertex
        x_vertex = signs * 5.0
        
        # 4. Evaluate vertex
        if self.calls < self.budget and time.time() < self.deadline:
            f_vertex = func(x_vertex)
            self.calls += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
                
        # 5. Use remaining budget for local polish (Nelder-Mead)
        # This handles any slight non-linearity or ensures convergence
        # if the vertex was not the absolute best due to some edge case.
        # However, for a linear function, the vertex IS the best.
        # We will run a simple coordinate descent or Nelder-Mead if budget remains.
        
        if self.calls < self.budget - 10 and time.time() < self.deadline:
            # Simple local search from best_x
            # Since it's linear, local search might just bounce.
            # But Nelder-Mead is robust.
            from scipy.optimize import minimize
            
            # Define a wrapper to count calls and respect deadline
            wrapper_calls = [0]
            def wrapper(x):
                if time.time() > self.deadline:
                    return np.inf
                wrapper_calls[0] += 1
                return func(x)
            
            # Start from best_x
            res = minimize(wrapper, best_x, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            # Check if result is better
            # Note: res.fun might be inf if deadline hit
            if res.fun < best_f and np.isfinite(res.fun):
                # Verify with a direct call if possible, or trust res
                # The wrapper already called func, so best_f should be updated
                # but we need to update best_x
                best_x = res.x
                best_f = res.fun
                self.calls += wrapper_calls[0] # Update global counter roughly

        return best_x, best_f
