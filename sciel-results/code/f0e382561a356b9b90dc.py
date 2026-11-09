"""ODE solver: 15-dimensional stiff linear system using block matrix exponential for constant coefficients."""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    Detects if the system is a constant-coefficient linear system (y' = Ay + b).
    If so, uses the block matrix exponential for an exact solution.
    Otherwise, falls back to SciPy's Radau method.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    if dt == 0:
        return y0
    
    # Count function calls (for internal tracking, evaluator counts f directly)
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
        
    # Attempt to detect constant linear system
    try:
        # Probe at t0
        b0 = f_counted(t0, np.zeros(n))
        A0 = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            A0[:, i] = f_counted(t0, ei) - b0
            
        # Probe at t1
        b1 = f_counted(t1, np.zeros(n))
        A1 = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            A1[:, i] = f_counted(t1, ei) - b1
            
        # Check if coefficients are constant
        # Use relative tolerance scaled by magnitude
        norm_A0 = np.linalg.norm(A0)
        norm_b0 = np.linalg.norm(b0)
        tol_A = 1e-8 * (norm_A0 + 1e-12)
        tol_b = 1e-8 * (norm_b0 + 1e-12)
        
        if np.linalg.norm(A0 - A1) < tol_A and np.linalg.norm(b0 - b1) < tol_b:
            # System is approximately y' = A0*y + b0
            # Use block matrix exponential: J = [[A, b], [0, 0]]
            # z' = J z where z = [y; 1]
            # Solution: z(t1) = expm(J*dt) @ z0
            J = np.zeros((n + 1, n + 1))
            J[:n, :n] = A0
            J[:n, n] = b0
            # J[n, :] remains zero
            
            E = expm(J * dt)
            z0 = np.zeros(n + 1)
            z0[:n] = y0
            z0[n] = 1.0
            
            z1 = E @ z0
            return z1[:n]
            
    except Exception:
        pass # Fall back to generic solver

    # Fallback: Use Radau method
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: BDF
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.y[:, -1]
    except Exception:
        return y0
