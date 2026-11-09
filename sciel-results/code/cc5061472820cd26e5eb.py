"""
Custom RKF45 ODE integrator for the Lotka-Volterra system.

Strategy:
- Implement a custom adaptive Runge-Kutta-Fehlberg (RKF45) method.
- Use very tight tolerances (rtol=1e-13, atol=1e-15) to force smaller steps.
- Enforce a minimum step size (h_min) to prevent step collapse and ensure
  sufficient resolution, potentially driving error below the 1e-14 floor.
- Track function calls and time to respect the 2,000,000 call limit and 300s deadline.
"""
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the given ODE system over t_span starting from y0.
    Returns the state vector at t = t_span[1].
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    
    # Edge case: zero-length span
    if t0 == t1:
        return y

    start_time = time.time()
    DEADLINE = 280.0  # seconds; safety margin
    MAX_CALLS = 2_000_000
    call_count = 0

    # RKF45 coefficients
    # Standard RKF45 coefficients (Fehlberg)
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1
    a6 = 1/2
    
    b21 = 1/4
    b31 = 3/32
    b32 = 9/32
    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197
    b51 = 439/216
    b52 = -8
    b53 = 3680/513
    b54 = -845/4104
    b61 = -8/27
    b62 = 2
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40
    
    # 5th order weights
    c1 = 25/216
    c2 = 0
    c3 = 1408/2565
    c4 = 2197/4104
    c5 = -1/5
    c6 = 0
    
    # 4th order weights (for error estimation)
    d1 = 16/135
    d2 = 0
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55

    rtol = 1e-13
    atol = 1e-15
    h_min = 1e-10
    h_max = 1e10
    safety = 0.9
    
    t = t0
    h = 0.01  # Initial step guess
    
    while t < t1:
        # Check time and call budget
        if time.time() - start_time > DEADLINE:
            break
        if call_count > MAX_CALLS:
            break
            
        # Ensure we don't step past t1
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
            
        # Compute stages
        k1 = f(t, y)
        call_count += 1
        
        k2 = f(t + a2*h, y + h*(b21*k1))
        call_count += 1
        
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        call_count += 1
        
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        call_count += 1
        
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        call_count += 1
        
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        call_count += 1
        
        # Compute 5th and 4th order solutions
        y5 = y + h*(c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
        y4 = y + h*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        
        # Error estimate
        err = np.max(np.abs(y5 - y4) / (atol + rtol * np.abs(y5)))
        
        if err <= 1.0:
            # Accept step
            t += h
            y = y5
            # Adjust step size
            if err == 0.0:
                h_new = h * 5.0
            else:
                h_new = h * safety * (err ** (-0.2))
            h = min(h_new, h_max)
        else:
            # Reject step, reduce h
            h = h * safety * (err ** (-0.2))
            if h < h_min:
                h = h_min
                
    return y
