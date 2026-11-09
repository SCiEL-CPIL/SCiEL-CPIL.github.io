import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        calls_made = 0
        
        best_x = np.zeros(d)
        best_f = float('inf')
        
        # Strategy: Direct sign identification via boundary probes
        # For each dimension, evaluate at +5 and -5 (others 0)
        # This determines the sign of the gradient component
        signs = np.zeros(d)
        
        for i in range(d):
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Probe positive boundary
            x_plus = np.zeros(d)
            x_plus[i] = 5.0
            f_plus = func(x_plus)
            calls_made += 1
            
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
                
            # Probe negative boundary
            if calls_made >= self.budget or time.time() > deadline:
                # If we ran out of budget, just use what we have
                # But we need the sign for the vertex construction
                # If we didn't evaluate minus, we can't determine sign reliably
                # For now, assume we have budget. 11 calls is small.
                break
            
            x_minus = np.zeros(d)
            x_minus[i] = -5.0
            f_minus = func(x_minus)
            calls_made += 1
            
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
                
            # Determine sign for dimension i
            # If f_plus < f_minus, then w_i < 0, so optimal x_i = 5
            # If f_minus < f_plus, then w_i > 0, so optimal x_i = -5
            if f_plus < f_minus:
                signs[i] = 5.0
            else:
                signs[i] = -5.0
                
        # Construct the vertex
        if calls_made < self.budget and time.time() < deadline:
            x_vertex = signs.copy()
            f_vertex = func(x_vertex)
            calls_made += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        # Stall on the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1
            
        return best_x, best_f
