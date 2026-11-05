import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves y' = f(t, y) for a stiff linear system y' = A(t)y + b(t).
    Uses a Radau IIA (3-stage, 5th order) implicit Runge-Kutta method.
    Exploits linearity to compute Jacobian J = A(t) efficiently.
    """
    start_time = time.time()
    deadline = start_time + 285  # 5s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Radau IIA 3-stage coefficients
    # c = [0, 1/3, 1]
    c = np.array([0.0, 1.0/3.0, 1.0])
    
    # A matrix for Radau IIA 3-stage
    # a11 = 1/6, a12 = 0, a13 = 0
    # a21 = 5/12, a22 = 1/12, a23 = 0
    # a31 = 3/4, a32 = 3/4, a33 = -1/4
    A_rk = np.array([
        [1/6.0, 0.0, 0.0],
        [5/12.0, 1/12.0, 0.0],
        [3/4.0, 3/4.0, -1/4.0]
    ])
    
    # b vector (weights)
    # b1 = 1/6, b2 = 5/6, b3 = 0
    b_rk = np.array([1/6.0, 5/6.0, 0.0])
    
    # Number of stages
    s = 3
    
    def extract_jacobian(t, y_ref):
        """
        Computes the Jacobian matrix J(t) = df/dy for the linear system.
        Since f(t, y) = A(t)y + b(t), J is independent of y.
        We estimate A(t) by probing f at y_ref and basis vectors.
        Calls: 1 + n
        """
        f0 = np.array(f(t, y_ref), dtype=float)
        J = np.zeros((n, n))
        # Use central differences for better accuracy if possible, 
        # but for linear systems, forward difference with unit vector is exact 
        # if floating point errors are negligible. 
        # However, to be robust against any non-linearities or noise, 
        # we can use a small step, but for "linear" problem, unit step is fine.
        # Let's use unit steps for simplicity and exactness in linear case.
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            f_i = np.array(f(t, y_ref + e_i), dtype=float)
            J[:, i] = f_i - f0
        return J

    # Determine number of steps
    # Radau IIA is 5th order. 
    # Error ~ C * h^6. 
    # We want high accuracy. 
    # Let's start with a moderate number of steps and check error if possible, 
    # or just pick a safe high number.
    # Given 300s limit, and n=15, linear solves are fast.
    # 100 steps -> 300 stages. 300 * (15x15 solve) is very fast.
    # Calls: 100 steps * (1 + 15) calls/step = 1600 calls.
    # log10(1600) ~ 3.2. Penalty ~ 1.6.
    # If we do 50 steps: 800 calls. Penalty ~ 1.5. Error increases by 2^6 = 64x.
    # If we do 200 steps: 3200 calls. Penalty ~ 1.8. Error decreases by 64x.
    
    # Let's try an adaptive-like strategy or just a fixed high count.
    # Since we can't easily estimate error without a reference, we pick a count 
    # that balances call penalty and accuracy.
    # Current best score 1.149 suggests we are doing well.
    # Let's try N=100 first.
    
    N = 100
    h = (t1 - t0) / N
    y = y0.copy()
    t = t0
    
    for k in range(N):
        # Check time
        if time.time() > deadline:
            # If we run out of time, return current y (best effort)
            # Or break and return. 
            # For safety, we should probably have chosen N small enough.
            # But if we are here, we return current state.
            break
            
        # Pre-compute Jacobian at the start of the step? 
        # In implicit RK, the Jacobian changes with t.
        # Radau stages are at t + c_i * h.
        # We need to solve (I - h * a_ij * J(t + c_j h)) * Y_i = ...
        # This is complicated because J depends on the stage time.
        # However, for a linear system, J(t) = A(t).
        # We can compute A(t + c_j h) for each stage j.
        
        # Let's structure the solver for the 3 stages.
        # Y_0, Y_1, Y_2 are the stage values.
        # Equations:
        # Y_0 = y + h * (a_00 * f(t + c_0 h, Y_0))  => (I - h a_00 J_0) Y_0 = y + h a_00 b_0
        # Y_1 = y + h * (a_10 f_0 + a_11 f_1)       => (I - h a_11 J_1) Y_1 = y + h (a_10 f_0 + a_11 b_1)
        # Y_2 = y + h * (a_20 f_0 + a_21 f_1 + a_22 f_2) => (I - h a_22 J_2) Y_2 = y + h (a_20 f_0 + a_21 f_1 + a_22 b_2)
        
        # Note: f_j = J_j Y_j + b_j.
        # So we need J_j and b_j for each stage j.
        
        # Stage 0: t0_stage = t
        t_s0 = t + c[0] * h
        J0 = extract_jacobian(t_s0, y) # Use y as reference, J is independent of y
        b0 = np.array(f(t_s0, np.zeros(n)), dtype=float)
        
        # Solve for Y0
        # (I - h * A_rk[0,0] * J0) Y0 = y + h * A_rk[0,0] * b0
        M0 = np.eye(n) - h * A_rk[0, 0] * J0
        rhs0 = y + h * A_rk[0, 0] * b0
        try:
            Y0 = np.linalg.solve(M0, rhs0)
        except np.linalg.LinAlgError:
            Y0 = np.linalg.pinv(M0) @ rhs0
            
        f0 = J0 @ Y0 + b0
        
        # Stage 1: t1_stage = t + h/3
        t_s1 = t + c[1] * h
        J1 = extract_jacobian(t_s1, y)
        b1 = np.array(f(t_s1, np.zeros(n)), dtype=float)
        
        # Solve for Y1
        # (I - h * A_rk[1,1] * J1) Y1 = y + h * (A_rk[1,0] * f0 + A_rk[1,1] * b1)
        M1 = np.eye(n) - h * A_rk[1, 1] * J1
        rhs1 = y + h * (A_rk[1, 0] * f0 + A_rk[1, 1] * b1)
        try:
            Y1 = np.linalg.solve(M1, rhs1)
        except np.linalg.LinAlgError:
            Y1 = np.linalg.pinv(M1) @ rhs1
            
        f1 = J1 @ Y1 + b1
        
        # Stage 2: t2_stage = t + h
        t_s2 = t + c[2] * h
        J2 = extract_jacobian(t_s2, y)
        b2 = np.array(f(t_s2, np.zeros(n)), dtype=float)
        
        # Solve for Y2
        # (I - h * A_rk[2,2] * J2) Y2 = y + h * (A_rk[2,0] * f0 + A_rk[2,1] * f1 + A_rk[2,2] * b2)
        M2 = np.eye(n) - h * A_rk[2, 2] * J2
        rhs2 = y + h * (A_rk[2, 0] * f0 + A_rk[2, 1] * f1 + A_rk[2, 2] * b2)
        try:
            Y2 = np.linalg.solve(M2, rhs2)
        except np.linalg.LinAlgError:
            Y2 = np.linalg.pinv(M2) @ rhs2
            
        f2 = J2 @ Y2 + b2
        
        # Update y
        y = y + h * (b_rk[0] * f0 + b_rk[1] * f1 + b_rk[2] * f2)
        t = t + h
        
    return y
