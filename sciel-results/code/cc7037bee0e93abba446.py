import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1
        
        # For linear function f(x) = a^T x + b, the minimum on [-5,5]^dim
        # is at the vertex where x_i = -5*sign(a_i).
        # We need to determine the sign of each a_i.
        # Evaluate f at x = 5*e_i for each i. Then f(5*e_i) - f(0) = 5*a_i.
        # If 5*a_i < 0 (i.e., f(5*e_i) < f(0)), then a_i < 0, so optimal x_i = +5.
        # If 5*a_i > 0 (i.e., f(5*e_i) > f(0)), then a_i > 0, so optimal x_i = -5.
        
        signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            
            x_plus = x_center.copy()
            x_plus[i] = 5.0
            
            f_plus = func(x_plus)
            calls_used += 1
            
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            
            if f_plus < f_center:
                signs[i] = -1.0  # a_i < 0, go to +5
            else:
                signs[i] = 1.0   # a_i > 0, go to -5
                
        # Construct the optimal vertex
        x_vertex = np.where(signs < 0, 5.0, -5.0)
        
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # For a linear function, this vertex IS the global minimum.
        # No further calls needed.
        
        return best_x, best_f
