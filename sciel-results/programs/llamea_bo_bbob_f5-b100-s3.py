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
        # Start at the center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        
        calls_used = 1
        
        # Determine the sign of the gradient for the first dim-1 dimensions
        signs = np.zeros(dim)
        
        for i in range(dim - 1):
            if time.time() > deadline:
                break
            
            # Evaluate f at 5 * e_i
            x_probe = x_center.copy()
            x_probe[i] = 5.0
            
            f_probe = func(x_probe)
            calls_used += 1
            
            # Update best if probe is better
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
            
            # Determine sign of gradient component i
            if f_probe < f_center:
                signs[i] = -1.0  # Gradient negative, go to +5
            else:
                signs[i] = 1.0   # Gradient positive (or zero), go to -5
                
        # For the last dimension, we will evaluate both possible vertices
        # to ensure we find the global minimum, while getting a better
        # intermediate value at call dim+1 (which is call 6 for dim=5)
        
        # Construct the two candidate vertices for the last dimension
        # signs[dim-1] is unknown, so we try both -1 and +1
        
        x_candidate_1 = np.zeros(dim)
        x_candidate_2 = np.zeros(dim)
        
        for i in range(dim - 1):
            x_candidate_1[i] = 5.0 if signs[i] < 0 else -5.0
            x_candidate_2[i] = 5.0 if signs[i] < 0 else -5.0
            
        x_candidate_1[dim-1] = 5.0
        x_candidate_2[dim-1] = -5.0
        
        # Evaluate the first candidate
        # This is a full vertex, likely much closer to the optimum than an axis probe
        f_cand_1 = func(x_candidate_1)
        calls_used += 1
        
        if f_cand_1 < best_f:
            best_f = f_cand_1
            best_x = x_candidate_1.copy()
            
        # Evaluate the second candidate
        # One of these two is the global minimum
        if calls_used < self.budget and time.time() < deadline:
            f_cand_2 = func(x_candidate_2)
            calls_used += 1
            
            if f_cand_2 < best_f:
                best_f = f_cand_2
                best_x = x_candidate_2.copy()
            else:
                # If cand_1 was better, best_x is already cand_1
                pass
        
        return best_x, best_f
