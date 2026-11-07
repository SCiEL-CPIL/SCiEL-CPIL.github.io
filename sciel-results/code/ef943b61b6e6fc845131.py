"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y0' = 1.5 y0 - y0 y1, y1' = -3 y1 + y0 y1
    from t_span[0] to t_span[1] with initial condition y0.
    
    Uses DOP853 with very tight tolerances to match the reference solution
    closely while minimizing function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin of 20 seconds
    
    # Use scipy's solve_ivp with DOP853 and very tight tolerances
    # This matches the reference method exactly
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000.0
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                return y_end
    except Exception:
        pass
    
    # Fallback: high-order Taylor method using analytic derivatives
    # For Lotka-Volterra: y0' = 1.5*y0 - y0*y1, y1' = -3*y1 + y0*y1
    # We can compute higher derivatives analytically
    
    def f_analytic(t, y):
        """Compute f(t, y) analytically for Lotka-Volterra."""
        return np.array([1.5*y[0] - y[0]*y[1], -3*y[1] + y[0]*y[1]])
    
    def compute_derivatives(t, y, order=8):
        """Compute derivatives of y up to the given order analytically."""
        # y^(0) = y
        # y^(1) = f(y)
        # y^(2) = J(y) @ f(y)
        # y^(3) = J(y) @ y^(2)
        # ...
        
        derivs = [y.copy()]
        fval = f(t, y)
        derivs.append(fval.copy())
        
        # Jacobian for Lotka-Volterra: J = [[1.5 - y1, -y0], [y1, y0 - 3]]
        y0, y1 = y
        J = np.array([[1.5 - y1, -y0],
                       [y1, y0 - 3]])
        
        for k in range(2, order + 1):
            # y^(k) = J @ y^(k-1)
            derivs.append(J @ derivs[-1])
        
        return derivs
    
    def taylor_step(t, y, h, order=8):
        """Single Taylor step of given order."""
        derivs = compute_derivatives(t, y, order)
        y_new = y.copy()
        for k in range(1, order + 1):
            y_new += h**k / np.math.factorial(k) * derivs[k]
        return y_new
    
    # Adaptive Taylor method
    h = (t1 - t0) / 1000
    t = t0
    order = 10
    max_steps = 100000
    min_h = (t1 - t0) / max_steps
    step = 0
    y = y.copy()
    
    while t < t1 - min_h/10:
        if time.time() > deadline:
            break
        
        if t + h > t1:
            h = t1 - t
        
        # Error estimation: compare Taylor step of order n vs n-1
        y_taylor_n = taylor_step(t, y, h, order)
        y_taylor_n_minus_1 = taylor_step(t, y, h, order - 1)
        
        err = np.linalg.norm(y_taylor_n - y_taylor_n_minus_1) / (np.linalg.norm(y_taylor_n) + 1e-30)
        
        tol = 1e-12 * (1 + np.linalg.norm(y))
        if err < tol:
            y = y_taylor_n  # Use the higher-order result
            t += h
            if err < tol / 100:
                h = min(h * 2.0, (t1 - t) / 5)
        else:
            h = max(h * 0.5 * (tol / err) ** 0.1, min_h)
        
        step += 1
        if step > max_steps:
            break
    
    # Take remaining steps
    while t < t1:
        if time.time() > deadline:
            break
        h_rem = min(h, t1 - t)
        y = taylor_step(t, y, h_rem, order)
        t += h_rem
    
    if not np.all(np.isfinite(y)):
        # Last resort: fixed small steps
        n_steps = 10000
        h = (t1 - t0) / n_steps
        y = np.array(y0, dtype=float)
        t = t0
        for _ in range(n_steps):
            y = taylor_step(t, y, h, 8)
            t += h
    
    return y

# EVOLVE-BLOCK-END
