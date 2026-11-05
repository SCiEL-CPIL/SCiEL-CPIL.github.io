"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    
    # Strategy: Identify the linear system parameters A and b.
    # Assume y' = A(t)y + b(t).
    # We sample at t0 to get A, b.
    # We verify constancy with 1 extra call at t1.
    
    def extract_params(t_val):
        """Extract A and b at time t_val using n+1 calls."""
        zero_vec = np.zeros(n, dtype=float)
        b_val = np.array(f(t_val, zero_vec), dtype=float)
        
        A_val = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            # f(t, e_i) = A e_i + b = col_i(A) + b
            # col_i(A) = f(t, e_i) - b
            col_i = np.array(f(t_val, e_i), dtype=float) - b_val
            A_val[:, i] = col_i
        return A_val, b_val

    # Sample at t0
    A0, b0 = extract_params(t0)
    
    # Verify constancy at t1 using y0
    # If constant, f(t1, y0) should be A0 @ y0 + b0
    y_pred = A0 @ y0 + b0
    y_actual = np.array(f(t1, y0), dtype=float)
    
    # Check if parameters are constant
    # Use a relative tolerance
    tol = 1e-6
    is_const = np.allclose(y_pred, y_actual, rtol=tol, atol=1e-12)
    
    if is_const:
        # Solve constant linear system y' = A0 y + b0 analytically
        # Augment system to handle non-invertible A:
        # z' = [A b; 0 0] z, z(0) = [y0; 1]
        A_aug = np.zeros((n + 1, n + 1), dtype=float)
        A_aug[:n, :n] = A0
        A_aug[:n, n] = b0
        # Last row is zeros
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0
        z0[n] = 1.0
        
        # Time step h = t1 - t0
        h = t1 - t0
        
        # Compute matrix exponential of h * A_aug
        M = scipy.linalg.expm(h * A_aug)
        
        z_end = M @ z0
        
        y_end = z_end[:n]
        
        # Safety check for finite values
        if not np.all(np.isfinite(y_end)):
            # Fallback to numerical solver
            return _solve_implicit(f, t_span, y0, A0)
            
        return y_end
    else:
        # System is time-varying or non-linear (though task says linear)
        # Use a robust stiff solver. We have already used (n+1)+1 calls.
        # Pass A0 as initial Jacobian guess.
        return _solve_implicit(f, t_span, y0, A0)

def _solve_implicit(f, t_span, y0, jac_guess=None):
    """Fallback solver for non-constant or uncertain cases."""
    t0, t1 = t_span
    # Use Radau or BDF for stiffness
    # max_step = (t1-t0)/10 to ensure some resolution
    # Provide Jacobian if available to speed up convergence
    if jac_guess is not None:
        def jac(t, y):
            return jac_guess
    else:
        jac = None
        
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            max_step=(t1 - t0) / 10.0,
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If Radau fails, try BDF
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=jac,
            max_step=(t1 - t0) / 10.0,
            rtol=1e-10, 
            atol=1e-12
        )
        return sol.y[:, -1]
    except Exception:
        # Last resort: simple RK4
        return _rk4(f, t_span, y0)
def _rk4(f, t_span, y0):
    """Simple RK4 fallback."""
    t0, t1 = t_span
    n = len(y0)
    y = np.array(y0, dtype=float)
    steps = 1000
    h = (t1 - t0) / steps
    for i in range(steps):
        t = t0 + i * h
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

# EVOLVE-BLOCK-END
