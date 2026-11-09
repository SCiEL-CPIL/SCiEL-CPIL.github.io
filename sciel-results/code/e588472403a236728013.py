# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    For a linear function f(x) = c + v^T x, the minimum over [-5,5]^d
    is at a vertex. We determine the optimal vertex by finding the sign
    of each gradient component using d+1 evaluations (origin + d axis probes),
    then evaluate that vertex.
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i, evaluate at 5*e_i to determine gradient component v_i.
       After each probe, update the best-known partial vertex and evaluate it
       to make early progress visible to the AOCC metric.
    3. Once all gradient components are known, construct and evaluate the full
       optimal vertex x*.
    4. Stall with the best point found for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Step 2: Probe each axis to recover gradient components
        v = np.zeros(d)
        v_known = np.zeros(d, dtype=bool)
        
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Evaluate at 5*e_i
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            
            # Update gradient component: v_i = (f(5*e_i) - f(0)) / 5
            v[i] = (fi - f0) / 5.0
            v_known[i] = True
            
            # Update best if this axis point is better
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
            
            # Construct partial optimal vertex using known gradient signs
            # For unknown dimensions, use 0 (contributes nothing to linear term)
            x_partial = np.zeros(d)
            for j in range(d):
                if v_known[j]:
                    if v[j] < 0:
                        x_partial[j] = 5.0
                    else:
                        x_partial[j] = -5.0
                else:
                    x_partial[j] = 0.0
            
            # Evaluate the partial vertex for early progress
            if calls_made < self.budget and time.time() < deadline:
                fp = func(x_partial)
                calls_made += 1
                if fp < f_best:
                    f_best = fp
                    x_best = x_partial.copy()
        
        # Step 3: If we have the complete gradient, evaluate the full optimal vertex
        if np.all(v_known) and calls_made < self.budget and time.time() < deadline:
            x_vertex = np.zeros(d)
            for i in range(d):
                if v[i] < 0:
                    x_vertex[i] = 5.0
                else:
                    x_vertex[i] = -5.0
            
            fv = func(x_vertex)
            calls_made += 1
            if fv < f_best:
                f_best = fv
                x_best = x_vertex.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
