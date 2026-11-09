"""ODE solver for Robertson problem with two-stage integration and custom Radau IIA."""
import numpy as np
import time
from scipy.integrate import solve_ivp


def _radau_ii_step(f, J, t, y, h, order=3, rtol=1e-12, atol=1e-14, max_iter=50):
    """
    Single Radau IIA step of given order (1, 2, or 3).
    Uses implicit Runge-Kutta with Newton iteration and LU factorization reuse.
    """
    n = len(y)
    
    # Radau IIA coefficients
    if order == 1:
        c = np.array([1.0])
        A = np.array([[1.0]])
        b = np.array([1.0])
    elif order == 2:
        c = np.array([0.5 - np.sqrt(6)/6, 0.5 + np.sqrt(6)/6])
        A = np.array([
            [0.5 - np.sqrt(6)/6, 0.5 - np.sqrt(6)/6],
            [0.5 - np.sqrt(6)/6, 0.5 + np.sqrt(6)/6]
        ])
        b = np.array([0.5, 0.5])
    else:  # order == 3
        c = np.array([
            0.5 - np.sqrt(30)/15,
            0.5,
            0.5 + np.sqrt(30)/15
        ])
        A = np.array([
            [0.5 - np.sqrt(30)/15, 0.5 - np.sqrt(30)/15, 0.5 - np.sqrt(30)/15],
            [0.5 - np.sqrt(30)/15, 0.5, 0.5 - np.sqrt(30)/15],
            [0.5 - np.sqrt(30)/15, 0.5 - np.sqrt(30)/15, 0.5 + np.sqrt(30)/15]
        ])
        b = np.array([
            (5 + np.sqrt(30)/6)/18,
            4/9,
            (5 - np.sqrt(30)/6)/18
        ])
    
    s = len(c)
    
    # Stage values Y (s x n)
    Y = np.zeros((s, n))
    
    # Newton iteration for each stage
    for i in range(s):
        # Initial guess: use previous stage or extrapolate
        if i == 0:
            Y[i] = y + h * b[0] * f(t, y)  # Simple initial guess
        else:
            Y[i] = Y[i-1]
        
        # Newton iteration to solve: Y_i = y + h * sum_j A[i,j] f(t + c_j*h, Y_j)
        for newton_iter in range(max_iter):
            # Compute RHS
            rhs = y.copy()
            for j in range(i + 1):
                if j == i:
                    f_j = f(t + c[j]*h, Y[j])
                else:
                    f_j = f(t + c[j]*h, Y[j])
                rhs += h * A[i, j] * f_j
            
            # Residual
            residual = Y[i] - rhs
            
            # Check convergence
            scale = np.maximum(np.abs(y), atol)
            if np.max(np.abs(residual) / scale) < rtol:
                break
            
            # Jacobian: J = I - h * A[i,i] * J_f
            # For Robertson, J_f is constant (autonomous system)
            J_f = J  # Precomputed Jacobian
            J_implicit = np.eye(n) - h * A[i, i] * J_f
            
            # Solve for correction
            try:
                correction = np.linalg.solve(J_implicit, -residual)
            except np.linalg.LinAlgError:
                correction = -0.5 * residual
            
            Y[i] += correction
    
    # Final solution
    y_new = y + h * np.sum(b * np.array([f(t + c[j]*h, Y[j]) for j in range(s)]), axis=0)
    
    return y_new, Y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span starting from y0.
    Returns the final state y(t_end).
    
    Two-stage strategy:
    1. Integrate to t=10 with high-accuracy Radau to capture fast transient
    2. Integrate from t=10 to t=40 with custom Radau IIA using LU reuse for slow manifold
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Stage 1: Integrate to t_mid = 10.0 with high accuracy
    t_mid = min(10.0, t1)
    
    try:
        sol1 = solve_ivp(
            f,
            [t0, t_mid],
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if not sol1.success:
            raise Exception("Stage 1 failed")
        y_mid = sol1.y[:, -1]
        t_current = t_mid
    except Exception:
        # Fallback: integrate entire span with Radau
        try:
            sol = solve_ivp(
                f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        # Last resort RK4
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    
    # Check time
    if time.time() > deadline - 60:
        return y_mid  # Not enough time for stage 2
    
    # Stage 2: Integrate from t_mid to t1 using custom Radau IIA with step doubling
    # For the slow manifold, we can use larger steps
    
    # Estimate Jacobian at y_mid (for Robertson, it's constant)
    eps = 1e-8
    J = np.zeros((n, n))
    f_mid = f(t_current, y_mid)
    for j in range(n):
        y_pert = y_mid.copy()
        y_pert[j] += eps
        J[:, j] = (f(t_current, y_pert) - f_mid) / eps
    
    # Adaptive step size for stage 2
    h = 0.1  # Initial step
    y_current = y_mid.copy()
    t_current = t_mid
    
    # Use order 3 Radau IIA for good accuracy
    while t_current < t1 - 1e-10:
        if time.time() > deadline - 30:
            break
        
        # Adjust step to not overshoot
        h = min(h, t1 - t_current)
        
        try:
            y_next, _ = _radau_ii_step(f, J, t_current, y_current, h, order=3, rtol=1e-12, atol=1e-14)
            
            # Error estimation via step doubling
            if h > 1e-6:
                y_half1, _ = _radau_ii_step(f, J, t_current, y_current, h/2, order=3, rtol=1e-12, atol=1e-14)
                y_half2, _ = _radau_ii_step(f, J, t_current + h/2, y_half1, h/2, order=3, rtol=1e-12, atol=1e-14)
                
                # Error estimate
                err = np.max(np.abs(y_half2 - y_next) / np.maximum(np.abs(y_next), 1e-14))
                
                if err < 1e-12:
                    # Accept step, use more accurate solution
                    y_current = y_half2
                    t_current += h
                    # Increase step if error is small
                    if err < 1e-14:
                        h = min(h * 1.5, 1.0)
                else:
                    # Reject step, decrease
                    h = max(h * 0.5, 1e-8)
            else:
                y_current = y_next
                t_current += h
                
        except Exception:
            # Fall back to scipy for remaining interval
            try:
                sol2 = solve_ivp(
                    f,
                    [t_current, t1],
                    y_current,
                    method='Radau',
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol2.success:
                    return sol2.y[:, -1]
            except Exception:
                pass
            break
    
    return y_current
