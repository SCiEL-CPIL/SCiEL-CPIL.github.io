"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    # Strategy: Try to exploit linearity and autonomy to use Matrix Exponential.
    # This reduces the problem to a single matrix exponential, which is highly accurate
    # and requires only O(N) function calls to construct the matrix (plus verification).
    
    try:
        # 1. Construct Jacobian/Operator A by probing f with basis vectors
        # Assuming f(t, y) = A @ y (autonomous and linear)
        A = np.zeros((n, n), dtype=float)
        
        # We need to be careful with the number of calls.
        # If N is small (100), this is cheap.
        for j in range(n):
            e = np.zeros(n, dtype=float)
            e[j] = 1.0
            # Use t0 for the probe
            A[:, j] = f(t0, e)
            
            if time.time() > deadline:
                break
        
        # 2. Verify Linearity and Autonomy
        # If these checks fail, we fall back to a general ODE solver.
        is_linear_autonomous = True
        tol = 1e-10
        
        if time.time() < deadline:
            # Check Autonomy: f(t1, y) == f(t0, y)
            # Use a random vector
            np.random.seed(42)
            y_rand = np.random.randn(n)
            f_t0 = f(t0, y_rand)
            f_t1 = f(t1, y_rand)
            if np.linalg.norm(f_t0 - f_t1) > tol * (1 + np.linalg.norm(f_t0)):
                is_linear_autonomous = False
                
        if is_linear_autonomous and time.time() < deadline:
            # Check Linearity: f(t0, y1+y2) == f(t0, y1) + f(t0, y2)
            y1 = np.random.randn(n)
            y2 = np.random.randn(n)
            f_sum = f(t0, y1 + y2)
            f_ind = f(t0, y1) + f(t0, y2)
            if np.linalg.norm(f_sum - f_ind) > tol * (1 + np.linalg.norm(f_ind)):
                is_linear_autonomous = False
                
            # Also check homogeneity: f(t0, 2y) == 2f(t0, y)
            f_2y = f(t0, 2 * y1)
            f_y = f(t0, y1)
            if np.linalg.norm(f_2y - 2 * f_y) > tol * (1 + np.linalg.norm(f_y)):
                is_linear_autonomous = False

        if is_linear_autonomous:
            # 3. Solve using Matrix Exponential
            # y(t1) = exp(A * (t1 - t0)) @ y0
            dt = t1 - t0
            # expm is very accurate for N=100
            M = scipy.linalg.expm(A * dt)
            y_final = M @ y0
            
            # Final sanity check: finite values
            if np.all(np.isfinite(y_final)):
                return y_final
            else:
                # Fallback if expm produced NaNs (unlikely for stable heat eq)
                pass

    except Exception:
        pass

    # Fallback: Robust Implicit Solver
    # Use Radau (implicit Runge-Kutta) which is good for stiff problems
    # Or BDF. Radau is generally more accurate for the same cost.
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12,
            jac=lambda t, y: A if 'A' in locals() else None # Pass A if we have it and it's linear
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Try BDF as last resort
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-10
            )
            if sol2.success:
                return sol2.y[:, -1]
    except Exception:
        pass

    # Last resort: Simple RK4 with small steps (should not happen)
    # This is just to ensure we return *something*
    try:
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        return y
    except Exception:
        return y0

# EVOLVE-BLOCK-END
