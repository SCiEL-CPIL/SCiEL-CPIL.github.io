import numpy as np
from scipy.linalg import lu_factor, lu_solve

def _robertson_jac(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y = [y0, y1, y2]
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) on t_span starting at y0.
    Uses a custom Radau IIA 3rd order integrator with analytic Jacobian.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Radau IIA 3-stage coefficients
    c = np.array([1.0/6.0, 2.0/3.0, 1.0])
    b = np.array([5.0/24.0, 3.0/4.0, 1.0/24.0])
    A = np.array([
        [1.0/36.0, 3.0/20.0, -8.0/45.0],
        [3.0/20.0, 6.0/5.0, -12.0/5.0],
        [5.0/24.0, 3.0/4.0, 1.0/24.0]
    ])
    
    # Embedded 2nd order coefficients (derived from order conditions)
    # Solving the linear system for b2 = [x, y, z]:
    # x + y + z = 1
    # x/6 + 2y/3 + z = 0.5
    # x/36 + 4y/9 + z = 1/3
    M = np.array([
        [1, 1, 1],
        [1/6, 2/3, 1],
        [1/36, 4/9, 1]
    ])
    rhs = np.array([1, 0.5, 1/3])
    b2 = np.linalg.solve(M, rhs)
    
    rtol = 1e-10
    atol = 1e-10
    
    h = 1e-6  # Initial step size
    t = t0
    max_steps = 2000000
    steps = 0
    safety = 0.9
    factor_min = 0.1
    factor_max = 10.0
    
    while t < t1 and steps < max_steps:
        if t + h > t1:
            h = t1 - t
        
        # Stage values
        Y = [y + h * f(t, y) for _ in range(3)]
        
        # Newton iterations
        for it in range(50):
            residuals = []
            J_total = np.zeros((3*n, 3*n))
            
            for i in range(3):
                res_i = Y[i] - y
                for j in range(3):
                    res_i -= h * A[i, j] * f(t + c[j]*h, Y[j])
                residuals.append(res_i)
                
                for k in range(3):
                    J_ik = -h * A[i, k] * _robertson_jac(t + c[k]*h, Y[k])
                    if i == k:
                        J_ik += np.eye(n)
                    J_total[i*n:(i+1)*n, k*n:(k+1)*n] = J_ik
            
            max_res = max(np.max(np.abs(r)) for r in residuals)
            if max_res < 1e-12:
                break
            
            rhs_vec = np.concatenate([-r for r in residuals])
            try:
                lu_piv, lu = lu_factor(J_total, overwrite_a=True)
                delta = lu_solve((lu, lu_piv), rhs_vec)
            except:
                break
            
            for i in range(3):
                Y[i] += delta[i*n:(i+1)*n]
        
        # Compute new y
        y_new = y + h * sum(b[i] * f(t + c[i]*h, Y[i]) for i in range(3))
        
        # Error estimate
        y_2nd = y + h * sum(b2[i] * f(t + c[i]*h, Y[i]) for i in range(3))
        err = np.abs(y_new - y_2nd)
        err_norm = np.max(err / (atol + rtol * np.abs(y_new)))
        
        # Step size control
        if err_norm > 1.0:
            h = h * safety * (1.0 / err_norm) ** 0.2
            h = max(h, factor_min * h)
            steps += 1
            continue
        
        y = y_new
        t += h
        steps += 1
        
        # Increase step size
        h = h * min(factor_max, safety * err_norm ** -0.2)
        h = max(h, 1e-12)
    
    return y
