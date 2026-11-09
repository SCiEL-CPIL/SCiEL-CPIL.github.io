import numpy as np
from scipy.linalg import expm_multiply
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Probe the system to see if it is linear with constant coefficients: y' = A*y + b
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = f(t0, e_j) - b
    # Since f(t, y) = A*y + b, f(t0, e_j) = A*e_j + b = A[:, j] + b
    # So A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float) - b
        
    # Check if the system is consistent with y' = A*y + b at a different point
    # Use a simple test point, e.g., y_test = ones
    y_test = np.ones(n)
    t_test = t0 + 0.5 * (t1 - t0)
    f_test = np.array(f(t_test, y_test), dtype=float)
    f_pred = A @ y_test + b
    
    # If the prediction is close, we assume constant linear system
    if np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-12):
        # System is y' = A*y + b with constant A, b
        # Solution: y(t) = e^{A(t-t0)} y0 + \int_{t0}^t e^{A(t-\tau)} b d\tau
        # Let w = A^{-1} b. Then \int e^{A(t-\tau)} b d\tau = (e^{A(t-t0)} - I) w
        # y(t) = e^{A(t-t0)} (y0 + w) - w
        
        dt = t1 - t0
        
        # Compute w = A^{-1} b
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b
            
        # v = y0 + w
        v = y0 + w
        
        # Compute e^{A*dt} * v using expm_multiply for efficiency and stability
        # This avoids forming the full matrix exponential
        y_end = expm_multiply(A * dt, v) - w
        
        return y_end

    else:
        # Fallback: Use Radau solver with Jacobian
        def jac(t, y):
            # Finite difference Jacobian
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(y[j]))
                J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
            return J

        sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac, rtol=1e-12, atol=1e-14)
        if not sol.success:
             sol = solve_ivp(f, t_span, y0, method='BDF', jac=jac, rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
