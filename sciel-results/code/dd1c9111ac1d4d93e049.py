"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t0 to t1 using a fixed-step Dormand-Prince 5(4) method.
    
    Parameters:
    f : callable, f(t, y) -> y'
    t_span : tuple (t0, t1)
    y0 : initial condition, float array of shape (3,)
    
    Returns:
    y_end : float array, the state at t1
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince 5(4) coefficients
    # This is the standard DP54 method with 6 stages (one embedded for error estimation, but we use fixed step)
    # For fixed step, we just use the 5th order update.
    # Coefficients from Dormand and Prince (1980)
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1
    a7 = 1
    
    b21 = 1/5
    b31 = 3/40
    b32 = 9/40
    b41 = 44/45
    b42 = -56/15
    b43 = 32/9
    b51 = 19372/6561
    b52 = -25360/2187
    b53 = 64448/6561
    b54 = -212/729
    b61 = 9017/3168
    b62 = -355/33
    b63 = 46732/5247
    b64 = 49/176
    b65 = -5103/18656
    
    # 5th order weights
    c1 = 35/384
    c2 = 0
    c3 = 500/1113
    c4 = 125/192
    c5 = -2187/6784
    c6 = 11/84
    c7 = 0
    
    # Number of steps - chosen to balance accuracy and call count
    # Each step uses 6 calls to f
    # With 1000 steps: 6000 calls
    # With 500 steps: 3000 calls
    # Let's use 1000 steps for good accuracy
    steps = 1000
    h = (t1 - t0) / steps
    
    t = t0
    for _ in range(steps):
        # Stage 1
        k1 = f(t, y)
        
        # Stage 2
        y2 = y + h * b21 * k1
        k2 = f(t + a2 * h, y2)
        
        # Stage 3
        y3 = y + h * (b31 * k1 + b32 * k2)
        k3 = f(t + a3 * h, y3)
        
        # Stage 4
        y4 = y + h * (b41 * k1 + b42 * k2 + b43 * k3)
        k4 = f(t + a4 * h, y4)
        
        # Stage 5
        y5 = y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
        k5 = f(t + a5 * h, y5)
        
        # Stage 6
        y6 = y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
        k6 = f(t + a6 * h, y6)
        
        # 5th order update
        y = y + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
        t += h
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            # If we get non-finite, we can't recover easily with fixed step
            # Return what we have, but this shouldn't happen for Lorenz with these steps
            break
    
    return y
# EVOLVE-BLOCK-END
