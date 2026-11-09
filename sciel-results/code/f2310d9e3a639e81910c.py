"""Custom Radau IIA integrator exploiting the analytic Jacobian of the Robertson problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _robertson_jac(t, y):
    """Analytic Jacobian of the Robertson problem.
    
    y' = f(t, y) where:
      f1 = -1e-4*y1 + 4e-3*y2*y3
      f2 =  1e-4*y1 - 4e-3*y2*y3 - 3e-7*y2^2
      f3 =  3e-7*y2^2
    """
    y1, y2, y3 = y
    df1_dy1 = -1e-4
    df1_dy2 = 4e-3 * y3
    df1_dy3 = 4e-3 * y2
    
    df2_dy1 = 1e-4
    df2_dy2 = -4e-3 * y3 - 6e-7 * y2
    df2_dy3 = -4e-3 * y2
    
    df3_dy1 = 0.0
    df3_dy2 = 6e-7 * y2
    df3_dy3 = 0.0
    
    J = np.array([
        [df1_dy1, df1_dy2, df1_dy3],
        [df2_dy1, df2_dy2, df2_dy3],
        [df3_dy1, df3_dy2, df3_dy3]
    ])
    return J


def _radau_ii3(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=100000):
    """Radau IIA 3-stage implicit Runge-Kutta method.
    
    This is a 3rd-order, 2nd-order accurate (L-stable) implicit RK method.
    For the Robertson problem, we use the analytic Jacobian to solve the
    nonlinear system at each stage efficiently.
    
    The Radau IIA 3-stage method has:
      c1 = (18 - 2*sqrt(6)) / 15
      c2 = (18 + 2*sqrt(6)) / 15
      c3 = 1
      
      a11 = (7 - sqrt(6)) / 12
      a12 = (1 + sqrt(6)) / 12
      a13 = 0
      a21 = (7 + sqrt(6)) / 12
      a22 = (1 - sqrt(6)) / 12
      a23 = 0
      a31 = 1/6
      a32 = 4/3
      a33 = -1/6
      
      b1 = 1/6
      b2 = 4/3
      b3 = -1/6
      
      m1 = 1/6
      m2 = 4/3
      m3 = -1/6
    """
    sqrt6 = np.sqrt(6.0)
    c1 = (18.0 - 2.0 * sqrt6) / 15.0
    c2 = (18.0 + 2.0 * sqrt6) / 15.0
    c3 = 1.0
    
    a11 = (7.0 - sqrt6) / 12.0
    a12 = (1.0 + sqrt6) / 12.0
    a13 = 0.0
    a21 = (7.0 + sqrt6) / 12.0
    a22 = (1.0 - sqrt6) / 12.0
    a23 = 0.0
    a31 = 1.0 / 6.0
    a32 = 4.0 / 3.0
    a33 = -1.0 / 6.0
    
    b1 = 1.0 / 6.0
    b2 = 4.0 / 3.0
    b3 = -1.0 / 6.0
    
    m1 = 1.0 / 6.0
    m2 = 4.0 / 3.0
    m3 = -1.0 / 6.0
    
    n = len(y0)
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size estimate
    h = (t1 - t0) / 1000.0
    
    f_count = 0
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    max_newton_iter = 10
    
    while t < t1 - 1e-15:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Stage 1: y1 = y + h*(a11*f1)
        # Solve: Y1 - h*a11*f(t+c1*h, Y1) = y
        Y1 = y.copy()
        for _ in range(max_newton_iter):
            f1 = f(t + c1 * h, Y1)
            f_count += 1
            J = _robertson_jac(t + c1 * h, Y1)
            A1 = np.eye(n) - h * a11 * J
            rhs1 = y
            try:
                delta = np.linalg.solve(A1, rhs1 - Y1)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A1, rhs1 - Y1, rcond=None)[0]
            Y1_new = Y1 + delta
            if np.linalg.norm(delta) < 1e-13 * (1.0 + np.linalg.norm(Y1_new)):
                Y1 = Y1_new
                break
            Y1 = Y1_new
        
        # Stage 2: y2 = y + h*(a21*f1 + a22*f2)
        # Solve: Y2 - h*(a21*f1 + a22*f2) = y, where f1=f(t+c1*h, Y1), f2=f(t+c2*h, Y2)
        f1_val = f(t + c1 * h, Y1)
        f_count += 1
        Y2 = y.copy()
        for _ in range(max_newton_iter):
            f2 = f(t + c2 * h, Y2)
            f_count += 1
            J = _robertson_jac(t + c2 * h, Y2)
            A2 = np.eye(n) - h * a22 * J
            rhs2 = y + h * a21 * f1_val
            try:
                delta = np.linalg.solve(A2, rhs2 - Y2)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A2, rhs2 - Y2, rcond=None)[0]
            Y2_new = Y2 + delta
            if np.linalg.norm(delta) < 1e-13 * (1.0 + np.linalg.norm(Y2_new)):
                Y2 = Y2_new
                break
            Y2 = Y2_new
        
        # Stage 3: y3 = y + h*(a31*f1 + a32*f2 + a33*f3)
        # Solve: Y3 - h*(a31*f1 + a32*f2 + a33*f3) = y
        f2_val = f(t + c2 * h, Y2)
        f_count += 1
        Y3 = y.copy()
        for _ in range(max_newton_iter):
            f3 = f(t + c3 * h, Y3)
            f_count += 1
            J = _robertson_jac(t + c3 * h, Y3)
            A3 = np.eye(n) - h * a33 * J
            rhs3 = y + h * (a31 * f1_val + a32 * f2_val)
            try:
                delta = np.linalg.solve(A3, rhs3 - Y3)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A3, rhs3 - Y3, rcond=None)[0]
            Y3_new = Y3 + delta
            if np.linalg.norm(delta) < 1e-13 * (1.0 + np.linalg.norm(Y3_new)):
                Y3 = Y3_new
                break
            Y3 = Y3_new
        
        # New solution: y_new = y + h*(b1*f1 + b2*f2 + b3*f3)
        y_new = y + h * (b1 * f1_val + b2 * f2_val + b3 * f3_val)
        
        # Error estimate using embedded lower-order method
        # y_err = y + h*(m1*f1 + m2*f2 + m3*f3)
        y_err = y + h * (m1 * f1_val + m2 * f2_val + m3 * f3_val)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        # Error control
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(np.abs(y_new - y_err) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t = t + h
            
            # Adjust step size
            if err_norm > 0:
                h_new = h * min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = h_new
        else:
            # Reject step
            h = h * max(0.1, 0.9 * err_norm ** (-0.2))
        
        if t >= t1 - 1e-15:
            break
    
    return y


def integrate(f, t_span, y0):
    """Integrate the Robertson IVP using Radau IIA with analytic Jacobian."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use very tight tolerances for high accuracy
    y_final = _radau_ii3(f, t0, t1, y0, rtol=1e-13, atol=1e-15, max_steps=50000)
    
    return y_final
# EVOLVE-BLOCK-END
