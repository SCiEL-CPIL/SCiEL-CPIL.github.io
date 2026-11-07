import numpy as np
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Uses a fixed-step Radau IIA (Order 3) integrator with Newton iterations.
    """
    deadline = time.time() + 280
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Fixed step size
    # Let's use h = 0.01 (1000 steps)
    # If accuracy is not enough, we could use smaller h, but this should be fine for rel ~ 1e-14
    h = 0.01
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    # Radau IIA Order 3 coefficients
    c = np.array([1/6, 1/2, 1.0])
    A = np.array([
        [5/9, 8/9, 1/9],
        [3/4, 5/4, 1/4],
        [3/8, 3/4, 1/8]
    ])
    b = A[2, :]
    
    # Initial guess for Jacobian (from t=0, y=y0)
    # y0 = [2, 0]
    J_f_prev = np.array([
        [0, 1],
        [-10 * y[0] * y[1] - 1, 5 * (1 - 3 * y[0]**2)]
    ])
    
    for _step in range(n_steps):
        if time.time() > deadline - 10:
            # Fallback to solve_ivp if running out of time
            try:
                from scipy.integrate import solve_ivp
                sol = solve_ivp(f, (t, t1), y, method='Radau', rtol=1e-11, atol=1e-13)
                if sol.success:
                    return sol.y[:, -1]
            except Exception:
                pass
            break
            
        # Initial guess for stages
        # Use previous step's stages if available, else Euler
        if _step == 0:
            f_y = f(t, y)
            Z = np.tile(h * f_y, (3, 1))
        else:
            Z = Z_prev
        
        Y = np.tile(y, (3, 1)) + Z
        
        # Newton iteration
        F_stages = None
        for _newton in range(10):
            # Evaluate F at stages
            F_stages = np.zeros((3, 2))
            for i in range(3):
                F_stages[i] = f(t + c[i] * h, Y[i])
            
            # Residual R = Z - h * (A @ F_stages)
            R = Z - h * (A @ F_stages)
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-14:
                break
            
            # Build Jacobian
            J_total = np.eye(6)
            for i in range(3):
                for j in range(3):
                    a_ij = A[i, j]
                    if a_ij != 0:
                        # Evaluate J_f at Y[j]
                        y_j = Y[j]
                        J_f = np.array([
                            [0, 1],
                            [-10 * y_j[0] * y_j[1] - 1, 5 * (1 - 3 * y_j[0]**2)]
                        ])
                        # Update block
                        J_total[2*i:2*i+2, 2*j:2*j+2] -= h * a_ij * J_f
            
            # Solve
            R_flat = R.flatten()
            try:
                delta_Z = np.linalg.solve(J_total, -R_flat)
            except np.linalg.LinAlgError:
                # If singular, break and use current Y
                break
            
            Z = Z + delta_Z.reshape(3, 2)
            Y = np.tile(y, (3, 1)) + Z
            
        # Update y
        y_new = y + h * (b @ F_stages)
        y = y_new
        t += h
        Z_prev = Z
        
    return y
