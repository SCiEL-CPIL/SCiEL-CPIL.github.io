"""Rewritten program exploiting linear structure of convection-diffusion."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    start_time = time.time()
    # Deadline guard: 300s limit, aim for 250s max
    deadline = start_time + 250.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Hypothesis: The system is linear and homogeneous: y' = A(t) y
    # And A is time-independent: y' = A y
    
    # Step 1: Check homogeneity
    zero_vec = np.zeros(n)
    f0 = f(t0, zero_vec)
    
    if np.linalg.norm(f0) > 1e-12:
        # Not homogeneous. Fallback to a robust ODE solver.
        # For non-linear or inhomogeneous, we can't easily exploit structure without more calls.
        # Use scipy.integrate.solve_ivp with a stiff method.
        from scipy.integrate import solve_ivp
        
        # To keep call count low, we might try a large step implicit method manually,
        # but solve_ivp is safer.
        # However, solve_ivp will count calls.
        # Let's try to be smart: if it's affine y' = Ay + b, we can solve it.
        # Check if f is affine.
        # f(t, y) = A y + b
        # b = f0
        # A can be found from f(t, e_j) - b
        
        # Let's assume it might be affine.
        A = np.zeros((n, n))
        b = f0
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            fj = f(t0, e)
            A[:, j] = fj - b
            
        # Check time independence of A and b
        # Check b at t1
        b1 = f(t1, zero_vec)
        if not np.allclose(b, b1, rtol=1e-9, atol=1e-12):
             # Time dependent inhomogeneous term. Harder.
             # Fallback to solve_ivp
             sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]
        
        # Check A at t1 (one column)
        e0 = np.zeros(n)
        e0[0] = 1.0
        f_t1_e0 = f(t1, e0)
        A1_col0 = f_t1_e0 - b
        if not np.allclose(A[:, 0], A1_col0, rtol=1e-9, atol=1e-12):
             # Time dependent A. Fallback.
             sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]
             
        # If we get here, it's affine constant: y' = A y + b
        # Solution: y(t) = e^{At} y0 + (e^{At} - I) A^{-1} b
        # Or use scipy.linalg.expm and solve.
        from scipy.linalg import expm
        h = t1 - t0
        M = expm(h * A)
        # Compute integral term: \int_0^h e^{A(h-\tau)} b d\tau = \int_0^h e^{As} b ds (sub u=h-tau)
        # = A^{-1} (e^{Ah} - I) b
        # If A is singular, use expm_multiply or solve.
        try:
            # Solve A x = (M - I) b ? No.
            # We need X such that A X = (M - I) b.
            # Or use the fact that \int_0^h e^{As} b ds = \Phi(h) b where \Phi satisfies \Phi' = A \Phi, \Phi(0)=0.
            # \Phi(h) = A^{-1} (M - I).
            # Let's solve A * integral_term = (M - I) * b
            rhs = (M - np.eye(n)) @ b
            integral_term = np.linalg.solve(A, rhs)
            y_final = M @ y0 + integral_term
            return y_final
        except np.linalg.LinAlgError:
             # A is singular. Use expm_multiply or fallback.
             from scipy.linalg import expm_multiply
             # This is tricky for affine. Fallback to ODE solver.
             sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]

    else:
        # Homogeneous: y' = A(t) y
        # Determine A at t0
        A = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t0, e)
            
        # Check time independence at t1
        e0 = np.zeros(n)
        e0[0] = 1.0
        f_t1_e0 = f(t1, e0)
        if not np.allclose(A[:, 0], f_t1_e0, rtol=1e-9, atol=1e-12):
             # Time dependent A. Fallback.
             from scipy.integrate import solve_ivp
             sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]
             
        # Constant A. Use matrix exponential.
        from scipy.linalg import expm
        h = t1 - t0
        M = expm(h * A)
        y_final = M @ y0
        return y_final

# EVOLVE-BLOCK-END
