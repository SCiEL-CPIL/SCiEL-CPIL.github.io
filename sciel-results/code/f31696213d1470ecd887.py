import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Choose a large number of steps to maximize accuracy while staying within call limits
    # Max calls allowed: 2,000,000. RK4 uses 4 calls per step.
    # Max steps: 500,000.
    # We use 250,000 steps (1,000,000 calls) to be safe on performance and stability margin.
    N = 250000
    h = (t1 - t0) / N
    
    y = y0.copy()
    t = t0
    
    # Local variables for speed
    h_half = h / 2.0
    h_sixth = h / 6.0
    h_half_k1_coeff = h_half
    h_half_k2_coeff = h_half
    h_k3_coeff = h
    
    for _ in range(N):
        k1 = f(t, y)
        k2 = f(t + h_half, y + h_half * k1)
        k3 = f(t + h_half, y + h_half * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h_sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
    return y
