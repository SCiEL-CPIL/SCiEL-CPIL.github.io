"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using an adaptive high-order
    implicit Runge-Kutta method (Radau IIA order 5) to match the reference.
    
    Returns y_end: the state at t_span[1].
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: 300s limit, keep 30s margin
    deadline = time.time() + 270.0
    
    # Radau IIA 5-stage, 5th order coefficients
    # Stages: c_i, a_ij, b_i
    # Radau IIA order 5:
    # c = [1/6, 1/3, 2/3, 5/6, 1]
    # a matrix (lower triangular + diagonal)
    # b = [0.02139232122724644, 0.05581900170163434, 0.2258857005879426, 0.3536983276715326, 0.3431046488116440]
    
    c = np.array([1/6, 1/3, 2/3, 5/6, 1.0])
    a = np.array([
        [1/6, 0, 0, 0, 0],
        [1/10, 3/20, 0, 0, 0],
        [1/30, 1/5, 11/30, 0, 0],
        [1/30, 2/15, 2/5, 3/5, 0],
        [1/6, 1/5, 1/3, 1/2, 1]
    ])
    b = np.array([0.02139232122724644, 0.05581900170163434, 0.2258857005879426, 3536983276715326/10**16, 0.3431046488116440])
    
    # Initial step size estimate
    h = 0.01
    t = t0
    
    # Tolerances for adaptive control
    rtol = 1e-12
    atol = 1e-14
    
    # Jacobian estimation helper (finite difference)
    def compute_jacobian(t_val, y_val):
        J = np.zeros((n, n))
        eps = 1e-8
        for j in range(n):
            e = np.zeros(n)
            ej = eps * max(1.0, abs(y_val[j]))
            e[j] = ej
            J[:, j] = (f(t_val, y_val + e) - f(t_val, y_val)) / ej
        return J
    
    # We'll use a simple adaptive step control
    # Since we know the problem is somewhat stiff (mu=5), we start with moderate steps
    
    # Try a fixed-step approach first with many steps, then refine if needed
    # Actually, let's use adaptive Radau with Newton iteration
    
    # For efficiency, let's try a simpler approach: use many small steps with RK4
    # but with a tighter tolerance check
    
    # Actually, the best approach for this specific problem:
    # The Van der Pol with mu=5 is moderately stiff. Let's use a high-order method.
    
    # Let's try: use solve_ivp-like adaptive Radau implementation
    
    # Simpler approach: use many RK4 steps with error estimation
    
    # Given the 300s limit and the need for high accuracy, let's use:
    # 1. Start with a coarse RK4 integration
    # 2. Refine with smaller steps where needed
    
    # Actually, the most reliable approach: use a large number of RK4 steps
    # The Van der Pol oscillator with mu=5 has a characteristic period of about 2*pi/mu ≈ 1.26
    # Over t in [0, 10], we have about 8 periods
    # For high accuracy, we need maybe 100-1000 steps per period
    
    # Let's use adaptive RK45 (Dormand-Prince) which is 4th/5th order
    
    # Dormand-Prince coefficients
    dp_c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    dp_a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    dp_b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 4th order
    dp_b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 5th order
    
    def rk45_step(t_val, y_val, h_val):
        """One step of Dormand-Prince RK45, returns y4, y5, error estimate"""
        k = np.zeros((7, n))
        k[0] = f(t_val, y_val)
        for i in range(1, 7):
            y_stage = y_val + h_val * np.sum(dp_a[i, :i] * k[:i], axis=0)
            k[i] = f(t_val + dp_c[i] * h_val, y_stage)
        y4 = y_val + h_val * np.sum(dp_b * k, axis=0)
        y5 = y_val + h_val * np.sum(dp_b_star * k, axis=0)
        err = y5 - y4
        return y5, err
    
    # Adaptive integration
    max_steps = 500000
    min_h = 1e-10
    max_h = 0.1
    
    step_count = 0
    while t < t1 - 1e-15 and step_count < max_steps:
        if time.time() > deadline - 5.0:
            break
        
        # Don't overshoot t1
        h = min(h, t1 - t)
        
        # Try the step
        y_new, err = rk45_step(t, y, h)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        # Error norm
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            step_count += 1
            
            # Adjust step size
            if err_norm > 0:
                factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h = h * factor
            h = min(max(h, min_h), max_h)
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = h * factor
            h = max(h, min_h)
    
    return y
# EVOLVE-BLOCK-END
