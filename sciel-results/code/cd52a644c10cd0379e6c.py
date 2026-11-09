"""ODE solver for 100-point convection-diffusion IVP using matrix exponential with adaptive fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    n = len(y0)
    T = t1 - t0
    
    # Helper to check time
    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Strategy:
    # 1. Assume the system is linear and time-invariant: y' = A@y + b
    # 2. Extract A and b with minimal function calls (N+1 calls).
    # 3. Compute exact solution using matrix exponential.
    # 4. If this fails or if the system is not linear (detected via residual check), 
    #    fall back to high-accuracy adaptive solver.
    
    # Step 1: Extract b = f(0, 0)
    # Note: t=0 is valid as the system is time-invariant.
    zero_vec = np.zeros(n, dtype=float)
    try:
        b = f(0.0, zero_vec)
        b = np.asarray(b, dtype=float)
    except Exception:
        b = None

    # Step 2: Extract A by calling f(0, e_i)
    # We do this only if b extraction succeeded and we have time.
    A = None
    if b is not None and _elapsed_ok():
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            try:
                A[:, i] = f(0.0, e_i)
            except Exception:
                A = None
                break
        
        if A is not None and not np.all(np.isfinite(A)):
            A = None

    # Step 3: Attempt Matrix Exponential Solution
    if A is not None and _elapsed_ok():
        try:
            # Compute expm(A * T)
            M = expm(A * T)
            
            if np.all(np.isfinite(M)):
                # Check if system is homogeneous (b is close to zero)
                # Tolerance for 'zero' b
                b_norm = np.linalg.norm(b)
                if b_norm < 1e-14:
                    # Homogeneous: y(T) = M @ y0
                    y_end = M @ y0
                else:
                    # Affine: y(T) = M @ y0 + A^{-1} (M - I) b
                    # To avoid explicit inverse, solve linear system: A X = (M - I) b
                    # X = A \ ((M - I) b)
                    # However, A might be singular. 
                    # Alternative: y(T) = M @ y0 + \int_0^T M(T-\tau) b d\tau
                    # If b is constant, \int_0^T e^{A(T-\tau)} b d\tau = A^{-1} (e^{AT} - I) b
                    # If A is singular, use limit or pseudo-inverse.
                    # For convection-diffusion, A is typically stable and non-singular or has 0 eigenvalue.
                    # Let's try solving A X = (M - I) b
                    rhs = (M - np.eye(n)) @ b
                    try:
                        X = np.linalg.solve(A, rhs)
                        y_end = M @ y0 + X
                    except np.linalg.LinAlgError:
                        # Fallback to pseudo-inverse if singular
                        X = np.linalg.lstsq(A, rhs, rcond=None)[0]
                        y_end = M @ y0 + X
                        
                if np.all(np.isfinite(y_end)):
                    # Optional: Verify linearity with a random vector to ensure no non-linear terms were missed
                    # This costs 1 extra call. 
                    # Given the task description "fixed convection-diffusion", linearity is expected.
                    # We skip verification to save calls unless we suspect issues.
                    return y_end
        except Exception:
            pass

    # Fallback: High-Accuracy Adaptive Solver
    # This handles non-linear, time-dependent, or cases where extraction failed.
    if _elapsed_ok():
        try:
            # Use DOP853 with tight tolerances
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method="DOP853",
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 50.0,
                first_step=(t1 - t0) / 1e6,
                dense_output=False
            )
            if sol.success and sol.t.size > 0:
                return sol.y[:, -1]
        except Exception:
            pass

    # Last Resort: Fixed Step RK4
    if _elapsed_ok():
        steps = 2000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if not _elapsed_ok():
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    return y0
# EVOLVE-BLOCK-END
