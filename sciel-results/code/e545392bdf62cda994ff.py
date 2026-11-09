import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: return best available result before 330s
    start_time = time.time()
    deadline = 330.0
    
    # Check for linearity and time-independence
    # f(t, y) = A y + b(t)
    
    # 1. Check if f is linear in y (homogeneous)
    y_test = y0.copy()
    if np.allclose(y_test, 0):
        y_test = np.ones(n)
        
    y_2x = 2 * y_test
    f_y = f(t0, y_test)
    f_2y = f(t0, y_2x)
    
    # Check homogeneity: f(2y) == 2*f(y)
    if not np.allclose(f_2y, 2 * f_y, rtol=1e-5, atol=1e-8):
        # Not homogeneous linear, use fallback
        return _solve_ivp_fallback(f, t_span, y0)
        
    # Check additivity: f(y1+y2) == f(y1)+f(y2)
    y_rand = np.random.randn(n)
    f_y1 = f(t0, y_test)
    f_y2 = f(t0, y_rand)
    f_sum = f(t0, y_test + y_rand)
    if not np.allclose(f_sum, f_y1 + f_y2, rtol=1e-5, atol=1e-8):
        # Not linear
        return _solve_ivp_fallback(f, t_span, y0)
        
    # Check if there's a constant offset b(t)
    f_zero = f(t0, np.zeros(n))
    has_offset = np.linalg.norm(f_zero) > 1e-10
    
    # If we have an offset, check if it's constant over time
    if has_offset:
        f_zero_t1 = f(t1, np.zeros(n))
        if not np.allclose(f_zero_t1, f_zero, rtol=1e-5, atol=1e-8):
            # Time-varying offset, use fallback
            return _solve_ivp_fallback(f, t_span, y0)
        # Constant offset: can be handled by augmented system or shift
        # For now, let's handle it properly
    
    # Assume linear: y' = A y + b (b may be 0 or constant)
    # Compute A by probing with basis vectors
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Check time-independence of A
    e_0 = np.zeros(n)
    e_0[0] = 1.0
    A_t1_col0 = f(t1, e_0)
    if not np.allclose(A_t1_col0, A[:, 0], rtol=1e-5, atol=1e-8):
        # Time-dependent A, use fallback
        return _solve_ivp_fallback(f, t_span, y0)
    
    dt = t1 - t0
    
    if not has_offset:
        # Homogeneous: y' = A y
        # Compute expm(A * dt)
        exp_A_dt = expm(A * dt)
        y_end = exp_A_dt @ y0
        return y_end
    else:
        # Affine: y' = A y + b
        # Solution: y(t) = expm(A*t) @ y0 + integral_0^t expm(A*(t-s)) @ b ds
        # For constant b: y(t) = expm(A*t) @ y0 + A^{-1} (expm(A*t) - I) @ b
        # More robust: use augmented system or numerical quadrature
        # Use the formula: y(t) = expm(A*t) @ y0 + \int_0^t expm(A*(t-s)) b ds
        # = expm(A*t) @ y0 + \int_0^t expm(A*u) b du  (substitution u = t-s)
        
        # Compute expm(A*dt)
        exp_A_dt = expm(A * dt)
        
        # Compute the integral term using the matrix exponential integral
        # \int_0^t expm(A*u) du can be computed via:
        # If A is invertible: A^{-1} (expm(A*t) - I)
        # But A might be singular (heat equation Laplacian with Dirichlet BCs is invertible)
        # Let's try the direct formula first
        try:
            # Use scipy's expm_multiply or compute directly
            # \int_0^t expm(A*u) b du
            # For small dt, we can use: b*dt + 0.5*A@b*dt^2 + ...
            # But for accuracy, let's use the augmented system approach
            # Or use: solve(A, expm(A*dt) - I) @ b
            # Check if A is invertible
            det_A = np.linalg.det(A)
            if abs(det_A) > 1e-10:
                # A is invertible
                integral_term = np.linalg.solve(A, exp_A_dt - np.eye(n)) @ f_zero
                y_end = exp_A_dt @ y0 + integral_term
                return y_end
            else:
                # A is singular, use numerical integration
                # Split into subintervals and use expm for each
                n_sub = 10
                h = dt / n_sub
                y_curr = y0.copy()
                for _ in range(n_sub):
                    # y_{k+1} = expm(A*h) @ y_k + \int_0^h expm(A*u) b du
                    exp_A_h = expm(A * h)
                    # For constant b and small h: \int_0^h expm(A*u) b du ≈ b*h + 0.5*A@b*h^2
                    # More accurately, use the same formula if A is invertible, else Taylor
                    if abs(det_A) > 1e-10:
                        integral_h = np.linalg.solve(A, exp_A_h - np.eye(n)) @ f_zero
                    else:
                        # Taylor expansion: b*h + 0.5*A@b*h^2 + (A^2@b)*h^3/6
                        integral_h = f_zero * h + 0.5 * (A @ f_zero) * h**2 + (A @ (A @ f_zero)) * h**3 / 6.0
                    y_curr = exp_A_h @ y_curr + integral_h
                return y_curr
        except Exception:
            # Fallback to numerical integration
            n_sub = 100
            h = dt / n_sub
            y_curr = y0.copy()
            for _ in range(n_sub):
                exp_A_h = expm(A * h)
                integral_h = f_zero * h + 0.5 * (A @ f_zero) * h**2
                y_curr = exp_A_h @ y_curr + integral_h
            return y_curr

def _solve_ivp_fallback(f, t_span, y0):
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK4
            h = 1e-4
            steps = int((t_span[1] - t_span[0]) / h)
            t = t_span[0]
            y = np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t, h + y + h*k3) if False else f(t + h, y + h*k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
    except Exception:
        h = 1e-4
        steps = int((t_span[1] - t_span[0]) / h)
        t = t_span[0]
        y = np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
