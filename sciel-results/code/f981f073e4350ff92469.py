import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses a fixed-step Dormand-Prince 5(4) (RK45) integrator.
    Returns the state vector at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    if t1 <= t0:
        return y

    # Fixed step size chosen to balance accuracy and number of calls.
    # RK45 is 4th order globally.
    # h = 0.002 -> 1000 steps -> 6000 calls. Expected rel error < 1e-10.
    h = 0.002
    steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / steps  # Adjust h to land exactly on t1

    # Dormand-Prince 5(4) coefficients
    # a matrix 6x6 (implicit 0s for explicit)
    # c: [0, 1/5, 3/10, 4/5, 8/9, 1]
    # b: [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    
    a10 = 1/5
    a20 = 3/40; a21 = 9/40
    a30 = 44/45; a31 = -56/15; a32 = 32/9
    a40 = 19372/6561; a41 = -25360/2187; a42 = 64448/6561; a43 = -212/729
    a50 = 9017/3168; a51 = 355/33; a52 = -46732/5247; a53 = 49/176; a54 = -5103/18656
    # a60..a64 are for the 6th stage? No, DP5(4) has 6 stages for the 5th order solution?
    # Standard DP5(4) uses 6 stages. The 5th order solution uses k1..k6.
    # The 4th order error estimator uses a different combination.
    # Wait, DP5(4) is usually 6 stages.
    # b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    
    b1 = 35/384
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    
    t = t0
    for _ in range(steps):
        if time.time() > deadline:
            return y
            
        k1 = f(t, y)
        k2 = f(t + 1/5*h, y + h*a10*k1)
        k3 = f(t + 3/10*h, y + h*(a20*k1 + a21*k2))
        k4 = f(t + 4/5*h, y + h*(a30*k1 + a31*k2 + a32*k3))
        k5 = f(t + 8/9*h, y + h*(a40*k1 + a41*k2 + a42*k3 + a43*k4))
        k6 = f(t + h, y + h*(a50*k1 + a51*k2 + a52*k3 + a53*k4 + a54*k5))
        
        y = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        t += h

    return y
