import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin
    
    # Radau IIA(3) coefficients
    sqrt6 = np.sqrt(6.0)
    b = np.array([
        (1 - sqrt6/3) / 6,
        1/3,
        (1 + sqrt6/3) / 6
    ])
    c = np.array([
        1/3 - sqrt6/6,
        1/3,
        2/3
    ])
    A_mat = np.array([
        [(4 - sqrt6)/18, (2 + sqrt6)/18, 0],
        [(1 - sqrt6)/9, 1/3, 0],
        [(1 - sqrt6)/6, 1/2, 1/3]
    ])
    
    # Determine number of steps
    # We want high accuracy. Radau 3 is O(h^4).
    # Let's try a few step counts and pick one that is fast and accurate enough.
    # Given the stiffness (A ~ 21^2 * tridiag), explicit methods need tiny h.
    # Implicit Radau can take larger h.
    # Let's start with 20 steps.
    steps = 20
    
    # 1. Compute Jacobian J at y0
    # Use finite differences
    J = np.zeros((n, n))
    h_fd = 1e-7
    y_base = y.copy()
    f0 = f(t0, y_base)
    
    for j in range(n):
        y_pert = y_base.copy()
        y_pert[j] += h_fd
        f_pert = f(t0, y_pert)
        J[:, j] = (f_pert - f0) / h_fd
    
    # 2. Build the 3n x 3n system matrix M
    # M_ij = delta_ij * I - h * A_mat[i,j] * J
    h = (t1 - t0) / steps
    M = np.zeros((3*n, 3*n))
    I_n = np.eye(n)
    
    for i in range(3):
        for j in range(3):
            if i == j:
                M[i*n:(i+1)*n, j*n:(j+1)*n] = I_n
            M[i*n:(i+1)*n, j*n:(j+1)*n] -= h * A_mat[i,j] * J
            
    # 3. Factor M
    # We can use np.linalg.solve in the loop, or factor once.
    # Since M is constant, we can use LU decomposition via scipy or just np.linalg.solve 
    # which does LU internally. For better performance, we can use scipy.linalg.lu_factor.
    try:
        from scipy.linalg import lu_factor, lu_solve
        lu_piv = lu_factor(M)
        use_scipy_lu = True
    except ImportError:
        use_scipy_lu = False
    
    # 4. Integration Loop
    t = t0
    for k in range(steps):
        if time.time() > deadline:
            break
            
        # RHS is [y, y, y]
        rhs = np.zeros(3*n)
        rhs[0:n] = y
        rhs[n:2*n] = y
        rhs[2*n:3*n] = y
        
        # Solve M Y_vec = rhs
        if use_scipy_lu:
            Y_vec = lu_solve(lu_piv, rhs)
        else:
            Y_vec = np.linalg.solve(M, rhs)
            
        # Extract Y1, Y2, Y3
        Y1 = Y_vec[0:n]
        Y2 = Y_vec[n:2*n]
        Y3 = Y_vec[2*n:3*n]
        
        # Evaluate f at stages
        t1_stage = t + c[0] * h
        t2_stage = t + c[1] * h
        t3_stage = t + c[2] * h
        
        F1 = f(t1_stage, Y1)
        F2 = f(t2_stage, Y2)
        F3 = f(t3_stage, Y3)
        
        # Update y
        y = y + h * (b[0] * F1 + b[1] * F2 + b[2] * F3)
        t += h
        
    return y
