"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    t = t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s
    
    # Radau IIA 3-stage coefficients
    # Stages: s1, s2, s3
    # a matrix (3x3)
    a = np.array([
        [0.6206197468723347, 0.0, 0.0],
        [0.1121159598156254, 0.6206197468723347, 0.0],
        [0.0672642971105217, 0.2198159560171427, 0.7129197468723347]
    ])
    b = np.array([0.0672642971105217, 0.2198159560171427, 0.7129197468723347])
    
    # Error estimation weights (using a lower-order embedding)
    # For Radau IIA, we can estimate error using the difference between
    # the full method and a simplified version
    # Simple error estimate: use the last stage residual
    c = np.array([0.6206197468723347, 0.7327357058327681, 1.0])
    
    # Adaptive step size
    h = (t1 - t0) / 1000.0  # initial step
    h_min = (t1 - t0) / 1e7
    h_max = (t1 - t0)
    
    rtol = 1e-10
    atol = 1e-12
    
    # Jacobian cache
    J_cache = None
    J_cache_t = None
    J_cache_y = None
    
    def compute_jacobian(t_val, y_val):
        """Compute Jacobian using finite differences."""
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            scale = 1e-7 * max(1.0, abs(y_val[j]))
            e[j] = scale
            J[:, j] = (f(t_val, y_val + e) - f(t_val, y_val)) / scale
        return J
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Cap step to not overshoot
        if t + h > t1:
            h = t1 - t
        
        # Try to take a step
        z = y.copy()
        fz = f(t, z)
        
        # Newton iteration for the implicit stages
        converged = False
        for _newton in range(15):
            # Stage 1
            s1 = z + h * (a[0,0] * f(t + c[0]*h, z + h*(a[0,0]*f(t + c[0]*h, z))))
            # This is getting complex; let's do proper Newton for Radau IIA
            
            # Actually, let's solve the system properly
            # Stage equations:
            # Y1 = y + h * a11 * f(t + c1*h, Y1)
            # Y2 = y + h * (a21*f(t+c1*h,Y1) + a22*f(t+c2*h,Y2))
            # Y3 = y + h * (a31*f(t+c1*h,Y1) + a32*f(t+c2*h,Y2) + a33*f(t+c3*h,Y3))
            
            # Solve for Y1, Y2, Y3 using Newton
            Y = np.zeros((3, n))
            Y[0] = z + h * a[0,0] * fz
            Y[1] = z + h * a[1,1] * fz
            Y[2] = z + h * a[2,2] * fz
            
            for _inner in range(10):
                # Compute f at each stage
                f1 = f(t + c[0]*h, Y[0])
                f2 = f(t + c[1]*h, Y[1])
                f3 = f(t + c[2]*h, Y[2])
                
                # Residuals
                R0 = Y[0] - z - h * a[0,0] * f1
                R1 = Y[1] - z - h * (a[1,0]*f1 + a[1,1]*f2)
                R2 = Y[2] - z - h * (a[2,0]*f1 + a[2,1]*f2 + a[2,2]*f3)
                
                res_norm = max(np.linalg.norm(R0), np.linalg.norm(R1), np.linalg.norm(R2))
                if res_norm < 1e-12 * (1 + np.linalg.norm(z)):
                    break
                
                # Build Jacobian of the system
                # Need J_f at each stage
                J1 = compute_jacobian(t + c[0]*h, Y[0])
                J2 = compute_jacobian(t + c[1]*h, Y[1])
                J3 = compute_jacobian(t + c[2]*h, Y[2])
                
                # Solve linear system for corrections
                # This is a 3n x 3n system, but we can solve sequentially
                # dY0 - h*a00*dF1 = -R0
                # dY1 - h*(a10*dF1 + a11*dF2) = -R1
                # dY2 - h*(a20*dF1 + a21*dF2 + a22*dF3) = -R2
                
                # dF1 = J1 @ dY1, etc.
                # This is complex; use a simpler approach: solve each stage sequentially
                
                # Solve for dY0: (I - h*a00*J1) dY0 = -R0
                A0 = np.eye(n) - h * a[0,0] * J1
                dY0 = np.linalg.solve(A0, -R0)
                
                # Solve for dY1: (I - h*a11*J2) dY1 = -R1 + h*a10*J1@dY0
                A1 = np.eye(n) - h * a[1,1] * J2
                rhs1 = -R1 + h * a[1,0] * (J1 @ dY0)
                dY1 = np.linalg.solve(A1, rhs1)
                
                # Solve for dY2: (I - h*a22*J3) dY2 = -R2 + h*(a20*J1@dY0 + a21*J2@dY1)
                A2 = np.eye(n) - h * a[2,2] * J3
                rhs2 = -R2 + h * (a[2,0] * (J1 @ dY0) + a[2,1] * (J2 @ dY1))
                dY2 = np.linalg.solve(A2, rhs2)
                
                Y[0] += dY0
                Y[1] += dY1
                Y[2] += dY2
            
            # Check convergence
            f1 = f(t + c[0]*h, Y[0])
            f2 = f(t + c[1]*h, Y[1])
            f3 = f(t + c[2]*h, Y[2])
            R0 = Y[0] - z - h * a[0,0] * f1
            R1 = Y[1] - z - h * (a[1,0]*f1 + a[1,1]*f2)
            R2 = Y[2] - z - h * (a[2,0]*f1 + a[2,1]*f2 + a[2,2]*f3)
            res_norm = max(np.linalg.norm(R0), np.linalg.norm(R1), np.linalg.norm(R2))
            if res_norm < 1e-12 * (1 + np.linalg.norm(z)):
                converged = True
                break
        
        if not converged:
            # Reduce step and retry
            h *= 0.5
            if h < h_min:
                break
            continue
        
        # New solution
        y_new = z + h * (b[0]*f1 + b[1]*f2 + b[2]*f3)
        
        # Error estimation: use the difference between stages
        err = np.linalg.norm(h * (a[2,2]*f3 - b[2]*f3))
        err_norm = err / (atol + rtol * np.maximum(1.0, np.abs(y_new)))
        
        # Step size control
        if err_norm < 1e-10:
            h *= 5.0
        elif err_norm > 1.0:
            h *= 0.8 * (1.0 / err_norm) ** 0.25
        else:
            h *= 1.4
        
        h = max(h_min, min(h, h_max))
        
        y = y_new
        t += h
        if t > t1:
            t = t1
    
    return y
# EVOLVE-BLOCK-END
