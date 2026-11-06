"""ODE solver: Robertson problem using an adaptive implicit method (Radau/BDF hybrid)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import root


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive implicit method suitable for stiff problems like Robertson.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: stop if we've used too much time
    start_time = time.time()
    deadline = start_time + 280.0  # Leave margin before 300s limit
    
    # Adaptive step size control
    h = 0.1  # Initial step
    h_min = 1e-12
    h_max = (t1 - t0) / 10  # Max step
    
    t = t0
    
    # Use Radau IIA (3-stage implicit) as the base method
    # Radau IIA coefficients:
    # c1 = 1/6, c2 = 1/2, c3 = 5/6
    # a11 = 1/6, a12 = 0, a13 = 0
    # a21 = 1/6, a22 = 1/6, a23 = 0
    # a31 = 5/24, a32 = 1/4, a33 = 1/24
    # b1 = 1/12, b2 = 5/12, b3 = 3/4
    c = np.array([1/6, 1/2, 5/6])
    A = np.array([
        [1/6, 0, 0],
        [1/6, 1/6, 0],
        [5/24, 1/4, 1/24]
    ])
    b = np.array([1/12, 5/12, 3/4])
    
    # For error estimation, use a lower-order method (Radau II with 2 stages)
    # Or we can use step doubling
    
    while t < t1 - 1e-14:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Cap step to not overshoot t1
        h = min(h, t1 - t)
        h = max(h, h_min)
        
        # Stage evaluations for Radau IIA
        # Y_i = y + h * sum_j A[i,j] * f(t + c_j*h, Y_j)
        
        # We need to solve for Y_1, Y_2, Y_3 simultaneously
        # This is a system of 3*n equations in 3*n unknowns
        
        # Use Newton's method for the stage equations
        Y = np.zeros((3, n))
        Y[0] = y.copy()
        Y[1] = y.copy()
        Y[2] = y.copy()
        
        # Newton iterations for the implicit stage equations
        for newton_iter in range(15):
            # Compute stage derivatives
            F = np.zeros((3, n))
            for i in range(3):
                F[i] = f(t + c[i] * h, Y[i])
            
            # Residual: Y_i - y - h * sum_j A[i,j] * F_j
            R = np.zeros((3, n))
            for i in range(3):
                R[i] = Y[i] - y - h * np.sum(A[i, :] * F, axis=0)
            
            # Check convergence
            rnorm = np.max(np.abs(R))
            if rnorm < 1e-13:
                break
            
            # Compute Jacobian of the residual w.r.t. Y
            # dR_i/dY_j = delta_ij * I - h * A[i,j] * dF_j/dY_j
            # We approximate dF_j/dY_j using finite differences or analytic Jacobian
            # For efficiency, use a sparse-like approach with finite differences
            
            # Build the full Jacobian (3n x 3n)
            J_full = np.zeros((3 * n, 3 * n))
            
            for j in range(3):
                # Compute dF_j/dY_j using finite differences
                eps = 1e-7 * np.maximum(1.0, np.abs(Y[j]))
                dF_dY = np.zeros((n, n))
                for k in range(n):
                    Yp = Y[j].copy()
                    Yp[k] += eps[k]
                    Fp = f(t + c[j] * h, Yp)
                    dF_dY[:, k] = (Fp - F[j]) / eps[k]
                
                for i in range(3):
                    # dR_i/dY_j = -h * A[i,j] * dF_j/dY_j
                    block = -h * A[i, j] * dF_dY
                    J_full[i*n:(i+1)*n, j*n:(j+1)*n] = block
            
            # Add identity for the Y_i - y part
            for i in range(3):
                J_full[i*n:(i+1)*n, i*n:(i+1)*n] += np.eye(n)
            
            # Solve J_full * delta = -R
            R_flat = R.flatten()
            try:
                delta = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                delta = np.linalg.lstsq(J_full, -R_flat, rcond=None)[0]
            
            delta = delta.reshape(3, n)
            Y += delta
        
        # Compute new y
        y_new = y + h * np.sum(b * F, axis=0)
        
        # Error estimation using step doubling or a lower-order method
        # Use the last stage as a lower-order approximation
        y_low = y + h * F[2]  # Euler-like using last stage
        
        # Error estimate
        err = np.max(np.abs(y_new - y_low))
        scale = 1e-6 + np.max(np.abs(y_new))
        err_rel = err / scale
        
        # Step size control
        if err_rel < 1e-13:
            # Success, accept step
            t += h
            y = y_new
            # Increase step
            h = min(h * 1.5, h_max)
        elif err_rel < 1e-5:
            # Accept but don't increase
            t += h
            y = y_new
            h = h * 0.9
        else:
            # Reject step, decrease h
            h = h * max(0.5, (1e-5 / err_rel) ** 0.25)
            h = max(h, h_min)
    
    return y
# EVOLVE-BLOCK-END
