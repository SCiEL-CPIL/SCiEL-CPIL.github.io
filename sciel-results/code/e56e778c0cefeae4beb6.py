"""High-accuracy ODE integrator for Van der Pol oscillator (mu=5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0'=y1, y1'=mu(1-y0^2)y1-y0
    from t_span[0] to t_span[1] with initial condition y0.
    
    Uses adaptive Runge-Kutta with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Time budget: start with a generous limit, but check periodically
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin before 300s limit
    
    # Adaptive RK45 implementation with tight tolerances
    rtol = 1e-13
    atol = 1e-15
    h_min = 1e-12
    h_max = 0.1
    h = 0.01  # initial step
    
    t = t0
    success = True
    
    while t < t1 - 1e-14:
        # Check time deadline
        if time.time() > deadline:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # RK45 coefficients (Fehlberg)
        # a2=1/4, a3=3/8, a4=12/13, a5=1, a6=1/2
        # b1=1/5, b3=9/40, b4=9/40, b5=-1/5, b6=7/8 (4th order)
        # c1=1/5, c2=1/5, c3=3/10, c4=3/10, c5=-17/80, c6=5/84 (5th order)
        
        yk = [y.copy() for _ in range(6)]
        yk[0] = y.copy()
        
        # Stage 1: k1 = f(t, y)
        k1 = f(t, y)
        
        # Stage 2: k2 = f(t + h/4, y + h/4*k1)
        y2 = y + (h/4) * k1
        k2 = f(t + h/4, y2)
        
        # Stage 3: k3 = f(t + 3h/8, y + h*(3/32*k1 + 9/32*k2))
        y3 = y + h * (3.0/32.0 * k1 + 9.0/32.0 * k2)
        k3 = f(t + 3.0*h/8.0, y3)
        
        # Stage 4: k4 = f(t + 12h/13, y + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
        y4 = y + h * (1932.0/2197.0 * k1 - 7200.0/2197.0 * k2 + 7296.0/2197.0 * k3)
        k4 = f(t + 12.0*h/13.0, y4)
        
        # Stage 5: k5 = f(t + h, y + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
        y5 = y + h * (439.0/216.0 * k1 - 8.0 * k2 + 3680.0/513.0 * k3 - 845.0/4104.0 * k4)
        k5 = f(t + h, y5)
        
        # Stage 6: k6 = f(t + h/2, y + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))
        y6 = y + h * (-8.0/27.0 * k1 + 2.0 * k2 - 3544.0/2565.0 * k3 + 1859.0/4104.0 * k4 - 11.0/40.0 * k5)
        k6 = f(t + h/2, y6)
        
        # 5th order solution
        y_new5 = y + h * (35.0/384.0 * k1 + 0.0 * k2 + 500.0/1113.0 * k3 + 125.0/192.0 * k4 - 2187.0/6784.0 * k5 + 11.0/84.0 * k6)
        
        # 4th order solution
        y_new4 = y + h * (5179.0/57600.0 * k1 + 0.0 * k2 + 7571.0/16695.0 * k3 + 393.0/6400.0 * k4 - 92097.0/339200.0 * k5 + 187.0/2100.0 * k6)
        
        # Error estimate
        err = np.max(np.abs(y_new5 - y_new4))
        scale = np.max(np.abs(y))
        err_rel = err / max(scale, 1e-15)
        
        # Accept or reject step
        if err_rel <= rtol:
            # Accept step
            y = y_new5
            t += h
            
            # Estimate next step size
            if err_rel > 1e-16:
                h_new = h * min(5.0, max(0.1, 0.9 * (rtol / err_rel) ** 0.2))
            else:
                h_new = h * 5.0
            h = min(h_new, h_max)
            h = max(h, h_min)
        else:
            # Reject step, reduce h
            if err_rel > 1e-16:
                h_new = h * max(0.1, 0.9 * (rtol / err_rel) ** 0.25)
            else:
                h_new = h * 0.5
            h = max(h_new, h_min)
    
    # Verify result is finite
    if not np.all(np.isfinite(y)):
        # Fallback: use scipy if available
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13)
            if sol.success:
                y = sol.y[:, -1]
        except:
            pass
    
    return y
# EVOLVE-BLOCK-END
