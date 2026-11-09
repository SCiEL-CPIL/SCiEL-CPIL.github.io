import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 10.0 # Safety margin for LP solve
    
    # Ensure correct shapes
    # prizes: (5, 100), weights: (5, 100)
    # Prize per item is sum of rows
    P = prizes.sum(axis=0) # (100,)
    W = weights # (5, 100)
    
    # Dual LP:
    # Minimize sum(lambda)
    # Subject to: W @ lambda >= P
    #             lambda >= 0
    
    c = np.ones(5)
    
    # Inequality constraints: A_ub @ x <= b_ub
    # We have W @ lambda >= P  =>  -W @ lambda <= -P
    A_ub = -W.T # (100, 5)
    b_ub = -P   # (100,)
    
    bounds = [(0, None)] * 5
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            lambdas = res.x
            # Reduced cost
            rc = P - W @ lambdas
            eta = np.maximum(rc, 0)
        else:
            # Fallback
            eta = P / (W.sum(axis=0) + 1e-9)
    except Exception:
        # Fallback
        eta = P / (W.sum(axis=0) + 1e-9)
        
    # Check time
    if time.time() > deadline:
         # If we are running out of time (unlikely for LP, but good practice)
         # Just return whatever we have or fallback
         pass
         
    return eta
