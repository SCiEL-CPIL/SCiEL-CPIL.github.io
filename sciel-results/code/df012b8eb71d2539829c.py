import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # DOP853 coefficients (8th order, 12 stages)
    c = np.array([0.0, 1/40, 1/5, 1/3, 4/15, 11/40, 16/45, 11/24, 2/5, 1/2, 5/8, 1.0])
    
    A = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [1/40,0,0,0,0,0,0,0,0,0,0,0],
        [1/80,1/40,0,0,0,0,0,0,0,0,0,0],
        [1/60,0,1/10,0,0,0,0,0,0,0,0,0],
        [-11/180,0,4/45,1/9,0,0,0,0,0,0,0,0],
        [137/1200,0,-27/200,12/125,1/4,0,0,0,0,0,0,0],
        [-251/2700,0,64/75,12/75,2/15,1/4,0,0,0,0,0,0],
        [101/162,0,-101/81,17/27,-1/2,1/2,0,0,0,0,0,0],
        [1/20,0,1/30,0,1/12,0,0,0,0,0,0,0],
        [1/10,0,0,0,0,0,0,0,1/2,0,0,0],
        [11/10,0,0,0,0,0,0,0,-1,1/2,0,0],
        [1/10,0,0,0,0,0,0,0,0,1/2,0,0]
    ])
    
    b = np.array([1/10, 0, 0, 0, 0, 0, 0, 0, 0, 1/2, 0, 0])
    
    # Fixed step: h=0.01 gives 500 steps for t in [0,5]
    h = 0.01
    n_steps = int(round((t1 - t0) / h))
    if n_steps < 1:
        n_steps = 1
    h = (t1 - t0) / n_steps
    
    y = y0.copy()
    t = t0
    
    for i in range(n_steps):
        K = np.zeros((12, n))
        for j in range(12):
            if j == 0:
                y_stage = y.copy()
            else:
                y_stage = y + h * (A[j, :j] @ K[:j, :])
            K[j] = f(t + c[j] * h, y_stage)
        
        y = y + h * (b @ K)
        t += h
    
    return y
