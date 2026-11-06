"""
Solves a 15-dimensional stiff linear ODE system y' = A y.
Exploits the linear constant-coefficient structure to compute the exact solution
via matrix exponential, minimizing function calls to ~15.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
import scipy.integrate as ssi

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Determine the system matrix A by probing the basis.
    # We assume f(t, y) = A y.
    A = np.zeros((n, n), dtype=float)
    
    # Evaluate f at standard basis vectors to get columns of A
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        # Check deadline
        if time.time() > deadline:
             sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]

    # Verify the model: Check if f(t1, y0) matches A @ y0
    f_check = f(t1, y0)
    y_model = A @ y0
    
    # Calculate relative difference
    diff = np.linalg.norm(f_check - y_model)
    scale = max(1.0, np.linalg.norm(y_model))
    
    if diff > 1e-8 * scale:
        # The system might be time-dependent or non-linear.
        # Fall back to a robust implicit solver.
        sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = ssi.solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]
            
    # 2. Compute the matrix exponential using Schur decomposition for better stability
    dt = t1 - t0
    try:
        # Use Schur decomposition based method for improved numerical stability
        T, Z = sla.schur(dt * A, output='complex')
        # Compute expm of the triangular matrix
        ET = sla.expm(T)
        # Transform back
        M = Z @ ET @ Z.conj().T
        # Take real part since the original system is real
        M = M.real
    except Exception:
        # Fallback to standard expm if Schur method fails
        try:
            M = sla.expm(dt * A)
        except Exception:
            # Last resort: use numerical integration
            sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]
        
    y_final = M @ y0
    
    # 3. Optional: Verify by checking if the solution satisfies the ODE at a midpoint
    # (This would cost additional function calls, so we skip it for efficiency)
    
    return y_final
# EVOLVE-BLOCK-END
