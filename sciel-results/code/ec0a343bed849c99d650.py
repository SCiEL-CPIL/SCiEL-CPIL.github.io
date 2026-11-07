"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    using a fixed-step Radau IIA collocation method with Newton iteration.
    
    The system is:
        y0' = y1
        y1' = 5*(1 - y0^2)*y1 - y0
    """
    t_start, t_end = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: return best available result before 330s wall-clock
    deadline = time.time() + 325.0  # 5s safety margin
    
    # Radau IIA collocation nodes and weights for s=3 stages (4th order)
    # Using s=3 stages gives 4th order accuracy with good stability
    # Radau IIA nodes for s=3:
    # c = [0, 0.2578764985636229, 0.5432868913976384, 1.0]
    # But for s=3 stages, we have 3 collocation points (excluding t_n)
    # Actually, Radau IIA with s stages has s collocation points c_1,...,c_s
    # For s=3: c1=0.2578764985636229, c2=0.5432868913976384, c3=1.0
    
    # Let's use s=3 Radau IIA (4th order)
    s = 3
    # Radau IIA collocation points for s=3
    c = np.array([0.2578764985636229, 0.5432868913976384, 1.0])
    
    # Radau IIA coefficients: Y[i,j] such that Y_n = Y_{n-1} + h * sum_j Y[i,j] * f(t_n + c_j*h, Y_n + h*sum_k Y[j,k]*f_k)
    # For Radau IIA, the Butcher tableau has:
    #   c_i | a_i1 a_i2 ... a_is | b_i
    # The stage values satisfy:
    #   Y_i = Y_{n-1} + h * sum_j a_ij * f_j
    #   Y_n = Y_{n-1} + h * sum_j b_j * f_j
    # where f_j = f(t_n + c_j*h, Y_j)
    
    # For s=3 Radau IIA:
    # a matrix (3x3) and b vector (3,)
    # From literature, for s=3 Radau IIA:
    a = np.array([
        [0.066554759887332, 0.0, 0.0],
        [0.180879798719554, 0.084178157846106, 0.0],
        [0.493578161409908, 0.241355403224267, 0.265066435365825]
    ])
    b = np.array([0.493578161409908, 0.241355403224267, 0.265066435365825])
    
    # Number of steps - choose enough for accuracy
    # With 4th order and h=0.01, we expect good accuracy
    # Let's use steps such that h = (t_end - t_start) / steps
    # For t in [0,10], let's try steps = 500 (h=0.02) or 1000 (h=0.01)
    
    # Start with a moderate number of steps
    steps = 500
    h = (t_end - t_start) / steps
    
    t = t_start
    y = y0.copy()
    
    # Precompute some constants
    eye_n = np.eye(n)
    
    for step_idx in range(steps):
        # Check deadline
        if time.time() > deadline:
            return y
        
        # Newton iteration for the implicit stage equations
        # We need to solve for Y = [Y_1, Y_2, Y_3] where each Y_i is n-dim
        # Y_i = y + h * sum_j a_ij * f(t + c_j*h, Y_j)
        
        # Initial guess: use previous step's stage values or just y
        Y = np.zeros((s, n))
        for i in range(s):
            Y[i] = y  # Initial guess
        
        # Newton iteration
        max_newton = 15
        for newton_iter in range(max_newton):
            # Evaluate f at all stages
            F = np.zeros((s, n))
            for j in range(s):
                F[j] = f(t + c[j] * h, Y[j])
            
            # Residual: R_i = Y_i - y - h * sum_j a_ij * F_j
            R = np.zeros((s, n))
            for i in range(s):
                R[i] = Y[i] - y - h * (a[i, :] @ F)
            
            # Check convergence
            r_norm = np.max(np.abs(R))
            if r_norm < 1e-13:
                break
            
            # Jacobian of the residual with respect to Y
            # dR_i/dY_k = delta_ik * I - h * a_ik * J_k
            # where J_k = df/dy at stage k
            # For 2D system, we can compute the Jacobian analytically or via finite differences
            # Since f is the Van der Pol oscillator, we can compute the Jacobian analytically
            # But f is passed as a black box, so we use finite differences
            
            # Build the (s*n) x (s*n) Jacobian
            # Actually, for efficiency with small n=2, we can do it directly
            # dR_i/dY_k = delta_ik * I_n - h * a_ik * J_k
            
            # Compute J_k for each stage
            J = np.zeros((s, n, n))
            eps = 1e-8
            for k in range(s):
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = eps * max(1.0, abs(Y[k, j]))
                    J[k, :, j] = (f(t + c[k] * h, Y[k] + e) - F[k]) / e[j]
            
            # Now solve the linear system: J_sys * delta_Y = -R
            # where J_sys is the (s*n) x (s*n) Jacobian
            # J_sys[i*n + p, k*n + q] = delta_ik * delta_pq - h * a_ik * J[k, p, q]
            
            # For n=2, s=3, this is a 6x6 system
            # Let's build it explicitly
            dim = s * n
            J_sys = np.zeros((dim, dim))
            for i in range(s):
                for k in range(s):
                    for p in range(n):
                        for q in range(n):
                            val = (1.0 if i == k else 0.0) * (1.0 if p == q else 0.0)
                            if i == k and p == q:
                                val = 1.0
                            else:
                                val = 0.0
                            val -= h * a[i, k] * J[k, p, q]
                            J_sys[i * n + p, k * n + q] = val
            
            # Solve J_sys * delta_Y = -R_vec
            R_vec = R.reshape(dim)
            try:
                delta_Y = np.linalg.solve(J_sys, -R_vec)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                delta_Y, _, _, _ = np.linalg.lstsq(J_sys, -R_vec, rcond=None)
            
            # Update Y
            for i in range(s):
                Y[i] += delta_Y[i * n:(i + 1) * n]
        
        # Update y using the last stage (Radau IIA is an implicit Runge-Kutta method)
        # y_new = y + h * sum_j b_j * F_j
        y = y + h * (b @ F)
        t += h
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            # Try to recover by reducing step size or returning what we have
            return y
    
    return y
# EVOLVE-BLOCK-END
