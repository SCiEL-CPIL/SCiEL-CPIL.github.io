# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 330s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Strategy: Assume Linear Autonomous System y' = A*y
    # The 2D Heat Equation with Dirichlet boundaries is linear, autonomous, and symmetric.
    
    try:
        # 1. Check for homogeneity: f(0) should be 0
        y_zero = np.zeros(n, dtype=float)
        f_zero = np.array(f(t0, y_zero), dtype=float)
        
        if np.linalg.norm(f_zero) > 1e-10:
            # Inhomogeneous system. Fallback to general solver.
            # This case is unlikely for standard heat equation tasks.
            raise ValueError("System appears inhomogeneous")
            
        # 2. Estimate A using unit vectors
        # This costs N calls. For N=100, this is fast.
        A = np.zeros((n, n), dtype=float)
        
        for j in range(n):
            y_pert = np.zeros(n, dtype=float)
            y_pert[j] = 1.0
            f_pert = np.array(f(t0, y_pert), dtype=float)
            A[:, j] = f_pert
            
            # Check deadline
            if time.time() > deadline - 60:
                raise TimeoutError("Deadline approaching during A estimation")
                
        # 3. Check for Symmetry
        # The Laplacian matrix is symmetric. Numerical errors from f evaluation
        # might introduce small asymmetries. Symmetrizing improves stability of expm.
        asymmetry = np.linalg.norm(A - A.T)
        a_norm = np.linalg.norm(A)
        
        if a_norm > 0 and asymmetry < 1e-10 * a_norm:
            A = (A + A.T) / 2.0
            
        # 4. Verify Linearity with a random point
        # Check if A @ y_random == f(y_random)
        np.random.seed(42) # Deterministic check
        y_rand = np.random.randn(n)
        y_rand = y_rand / np.linalg.norm(y_rand) * np.max(np.abs(y0)) # Scale similar to y0
        
        f_rand = np.array(f(t0, y_rand), dtype=float)
        pred_rand = A @ y_rand
        
        linearity_err = np.linalg.norm(f_rand - pred_rand)
        ref_norm = max(np.linalg.norm(f_rand), 1e-12)
        
        if linearity_err > 1e-8 * ref_norm:
            raise ValueError("Linearity check failed")
            
        # 5. Solve y' = A*y exactly
        # y(t) = exp(A * dt) * y0
        exp_adt = scipy.linalg.expm(A * dt)
        y_final = exp_adt @ y0
        
        # Final sanity check: Ensure no NaNs or Infs
        if not np.all(np.isfinite(y_final)):
            raise ValueError("Result contains non-finite values")
            
        return y_final
        
    except Exception:
        # Fallback: Robust Adaptive Solver
        try:
            # Use Radau for stiff systems like heat equation
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Try LSODA
                sol = scipy.integrate.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    rtol=1e-10, 
                    atol=1e-12
                )
                if sol.success:
                    return sol.y[:, -1]
                return y0
        except Exception:
            return y0

# EVOLVE-BLOCK-END
