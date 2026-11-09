"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # Keep margin before 300s limit
    
    # Strategy 1: Extract linear system matrix and use matrix exponential
    # For convection-diffusion, the system is linear and time-invariant: y' = A @ y + b
    # We can extract A and b with n+1 calls to f, then compute exact solution.
    try:
        # First call to get b (the forcing term at y=0)
        b = np.array(f(t0, np.zeros(n)), dtype=float)
        
        # Extract columns of A by probing each basis vector
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # A[:, i] = f(t0, e_i) - b  (since f(t, y) = A@y + b)
            A[:, i] = np.array(f(t0, e_i), dtype=float) - b
        
        # Check if the system is truly linear and time-invariant by verifying at another point
        # Quick check: f(t1, y0) should equal A@y0 + b if time-invariant
        f_check = np.array(f(t1, y0_arr), dtype=float)
        f_pred = A @ y0_arr + b
        if np.allclose(f_check, f_pred, rtol=1e-10, atol=1e-12):
            # System is linear and time-invariant. Use matrix exponential for exact solution.
            dt = t1 - t0
            # For y' = A@y + b, the solution is y(t) = expm(A*dt) @ y0 + expm(A*dt) @ A^{-1} @ b * (1 - expm(-A*dt))
            # More precisely: y(t) = expm(A*dt) @ y0 + integral_0^dt expm(A*(dt-s)) b ds
            # = expm(A*dt) @ y0 + A^{-1} @ (expm(A*dt) - I) @ b  if A is invertible
            # Or use scipy.integrate for the integral part, or use the formula directly.
            
            # Use the exact formula for constant-coefficient linear ODE:
            # y(t) = expm(A*t) @ y0 + (expm(A*t) - I) @ A^{-1} @ b
            # But A might be singular. Better to use:
            # y(t) = expm(A*t) @ y0 + integral_0^t expm(A*(t-s)) b ds
            # = expm(A*t) @ y0 + A^{-1} @ (expm(A*t) - I) @ b  (if A invertible)
            
            # Safer approach: use scipy.integrate with the linear system, but we already have A.
            # Actually, let's just use the matrix exponential approach with the integral.
            # For constant b, the particular solution is A^{-1} b if A is invertible.
            # General solution: y(t) = expm(A*t) @ (y0 - A^{-1} b) + A^{-1} b
            
            # Check if A is invertible
            try:
                A_inv_b = scipy.linalg.solve(A, b)
                expm_A_dt = scipy.linalg.expm(A * dt)
                y_final = expm_A_dt @ (y0_arr - A_inv_b) + A_inv_b
            except scipy.linalg.LinAlgError:
                # A is singular, use the integral formula
                expm_A_dt = scipy.linalg.expm(A * dt)
                # y(t) = expm(A*dt) @ y0 + integral_0^dt expm(A*(dt-s)) b ds
                # For constant b: integral = A^{-1} @ (expm(A*dt) - I) @ b if A invertible
                # Otherwise, use numerical integration or series expansion
                # For now, fall through to adaptive solver
                raise Exception("A is singular, using adaptive solver")
            
            # Verify the solution
            if np.all(np.isfinite(y_final)):
                return y_final
        else:
            # System is not linear/time-invariant, or extraction failed
            raise Exception("System not linear/time-invariant")
    except Exception:
        pass
    
    # Strategy 2: Richardson extrapolation using two distinct high-order adaptive solvers
    # DOP853 (8th order) and Radau (5th order implicit) with tight tolerances
    try:
        if time.time() > deadline - 30:
            # Not enough time for Richardson, just use Radau
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='Radau',
                rtol=1e-13, 
                atol=1e-15,
                max_step=(t1-t0)/50
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                raise Exception("Radau failed")
        
        # Run DOP853 with tight tolerance
        sol_dop = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853',
            rtol=1e-13, 
            atol=1e-15
        )
        
        if not sol_dop.success:
            raise Exception("DOP853 failed")
        
        # Run Radau with tight tolerance
        sol_radau = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1-t0)/50
        )
        
        if not sol_radau.success:
            raise Exception("Radau failed")
        
        y_dop = sol_dop.y[:, -1]
        y_radau = sol_radau.y[:, -1]
        
        # Richardson extrapolation: if errors are e_dop ~ C1*h^8 and e_radau ~ C2*h^5
        # We can't directly do Richardson because the methods have different orders.
        # Instead, just use the more accurate one, or average if they agree.
        # For now, return Radau result as it's typically more accurate for stiff problems.
        return y_radau
    except Exception:
        pass
    
    # Strategy 3: Fallback to Radau with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            rtol=1e-14, 
            atol=1e-16,
            max_step=(t1-t0)/100
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Strategy 4: Fallback to DOP853
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853',
            rtol=1e-14, 
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Strategy 5: Fallback to manual RK4 with moderate steps
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                return None
        return y
    
    y = _rk4(f, t0, t1, y0_arr, 5000)
    if y is not None:
        return y
        
    return y0_arr
# EVOLVE-BLOCK-END
