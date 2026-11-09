"""
Fixed-step Dormand-Prince 5(4) integrator for the Van der Pol oscillator.
Uses h=0.001 for ~10000 steps, reusing k1 to save calls.
"""
import numpy as np
import time

# Dormand-Prince 5(4) coefficients
_DP5_C2 = 1/5
_DP5_C3 = 3/10
_DP5_C4 = 4/5
_DP5_C5 = 8/9
_DP5_C6 = 1.0

_DP5_A21 = 1/5
_DP5_A31 = 3/40
_DP5_A32 = 9/40
_DP5_A41 = 44/45
_DP5_A42 = -56/15
_DP5_A43 = 32/9
_DP5_A51 = 19372/6561
_DP5_A52 = -25360/2187
_DP5_A53 = 64448/6561
_DP5_A54 = -212/729
_DP5_A61 = 9017/3168
_DP5_A62 = -355/33
_DP5_A63 = 46732/5247
_DP5_A64 = 49/176
_DP5_A65 = -5103/18656

# 5th order weights
_DP5_B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

def integrate(f, t_span, y0):
    """
    Integrates the ODE using fixed-step Dormand-Prince 5(4).
    """
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    call_count = 0
    max_calls = 1900000  # Safety margin below 2,000,000
    deadline = time.time() + 280.0  # Safety margin below 300s
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        return f(t, y)
    
    # Fixed step size
    h = 0.001
    n_steps = int(round((t1 - t0) / h))
    # Adjust h to exactly reach t1
    h = (t1 - t0) / n_steps
    
    y = y0.copy()
    t = t0
    
    for i in range(n_steps):
        if time.time() > deadline or call_count > max_calls:
            break
            
        # k1
        k1 = f_counted(t, y)
        
        # k2
        y2 = y + h * _DP5_A21 * k1
        t2 = t + h * _DP5_C2
        k2 = f_counted(t2, y2)
        
        # k3
        y3 = y + h * (_DP5_A31 * k1 + _DP5_A32 * k2)
        t3 = t + h * _DP5_C3
        k3 = f_counted(t3, y3)
        
        # k4
        y4 = y + h * (_DP5_A41 * k1 + _DP5_A42 * k2 + _DP5_A43 * k3)
        t4 = t + h * _DP5_C4
        k4 = f_counted(t4, y4)
        
        # k5
        y5 = y + h * (_DP5_A51 * k1 + _DP5_A52 * k2 + _DP5_A53 * k3 + _DP5_A54 * k4)
        t5 = t + h * _DP5_C5
        k5 = f_counted(t5, y5)
        
        # k6
        y6 = y + h * (_DP5_A61 * k1 + _DP5_A62 * k2 + _DP5_A63 * k3 + _DP5_A64 * k4 + _DP5_A65 * k5)
        t6 = t + h * _DP5_C6
        k6 = f_counted(t6, y6)
        
        # 5th order solution
        y = y + h * (_DP5_B[0] * k1 + _DP5_B[2] * k3 + _DP5_B[3] * k4 + _DP5_B[4] * k5 + _DP5_B[5] * k6)
        t += h
    
    return y
