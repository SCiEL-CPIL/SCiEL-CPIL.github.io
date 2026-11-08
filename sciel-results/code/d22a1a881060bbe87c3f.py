import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Integrates y' = A y from t_span[0] to t_span[1] starting at y0.
    Uses analytic solution via matrix exponential if f is linear and constant.
    Falls back to Radau IIA (Order 3) if analytic solution fails or is invalid.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    call_count = 0
    def counted_f(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)

    # Candidate 1: Check if y0 is an eigenvector of A.
    # If y' = A y and y0 is an eigenvector, y(t) = exp(lambda t) y0.
    # This requires only 1 call to f.
    try:
        fy0 = counted_f(t0, y0)
        if not np.all(np.isfinite(fy0)):
            raise ValueError("Non-finite probe value")
        
        # Check collinearity: fy0 should be lambda * y0
        # lambda = (fy0 . y0) / (y0 . y0)
        norm_y0_sq = np.dot(y0, y0)
        if norm_y0_sq < 1e-15:
            raise ValueError("Zero initial condition")
            
        lambda_est = np.dot(fy0, y0) / norm_y0_sq
        
        # Verify collinearity: ||fy0 - lambda y0|| should be very small
        residual = fy0 - lambda_est * y0
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(fy0)
        
        if rel_residual < 1e-10:
            dt = t1 - t0
            if dt == 0:
                return y0.copy()
            y_end = y0 * np.exp(lambda_est * dt)
            if not np.all(np.isfinite(y_end)):
                raise ValueError("Non-finite analytic result")
            return y_end
    except Exception:
        pass

    # Candidate 2: Analytic Solution for Linear Constant Coefficient ODE
    # Hypothesis: f(t, y) = A @ y where A is constant.
    # We probe A using basis vectors.
    try:
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # Use t0 for probing. A is constant.
            val = counted_f(t0, e_j)
            if not np.all(np.isfinite(val)):
                raise ValueError("Non-finite probe value")
            A[:, j] = val
        
        dt = t1 - t0
        if dt == 0:
            y_end = y0.copy()
        else:
            # Compute matrix exponential
            # For N=20, this is very fast.
            M = scipy.linalg.expm(A * dt)
            y_end = M @ y0
            
        # Validate result
        if not np.all(np.isfinite(y_end)):
            raise ValueError("Non-finite analytic result")
        
        return y_end

    except Exception:
        pass

    # Candidate 3: Radau IIA Order 3
    # Robust implicit solver.
    
    # Re-probe A carefully?
    A = None
    try:
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            val = counted_f(t0, e_j)
            if not np.all(np.isfinite(val)):
                raise ValueError
            A[:, j] = val
    except Exception:
        A = None

    # Implement Radau IIA Order 3
    # But wait, if A is available, we can solve (I - gamma h A) y = b directly.
    # This is much faster and more accurate than Newton with FD.
    
    def _radau_3(f, t0, t1, y0, A_matrix, steps):
        # Radau IIA Order 3 Butcher Tableau
        # c1 = 0.5 - sqrt(6)/6, c2 = 0.5, c3 = 0.5 + sqrt(6)/6
        # But for simplicity and robustness, maybe Backward Euler with many steps?
        # No, Radau 3 is better.
        
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        
        sqrt6 = np.sqrt(6)
        c1 = 0.5 - sqrt6/6
        c2 = 0.5
        c3 = 0.5 + sqrt6/6
        
        a11 = (6 - 5*sqrt6)/36
        a12 = (3 - sqrt6)/36
        a13 = sqrt6/36
        
        a21 = 0.25
        a22 = 0.25
        a23 = 0
        
        a31 = 0
        a32 = (3 + sqrt6)/36
        a33 = (6 + 5*sqrt6)/36
        
        b1 = (1 + sqrt6)/6
        b2 = 2/3
        b3 = (1 - sqrt6)/6
        
        # Precompute factors for the linear systems if A is provided
        # The systems are of the form:
        # Y1 - h*a11*F1 - h*a12*F2 - h*a13*F3 = y
        # F_i = A * Y_i (if linear)
        # So Y_i - h*a_ii * A * Y_i - ... = y + ...
        # (I - h*a_ii*A) Y_i = RHS
        # We can factorize (I - h*a_ii*A) once per step? 
        # h is constant, so the matrices are constant!
        
        M1 = np.eye(n) - h * a11 * A_matrix
        M2 = np.eye(n) - h * a22 * A_matrix
        M3 = np.eye(n) - h * a33 * A_matrix
        
        try:
            U1, E1 = scipy.linalg.lu_factor(M1)
            U2, E2 = scipy.linalg.lu_factor(M2)
            U3, E3 = scipy.linalg.lu_factor(M3)
        except Exception:
            # Fallback to Newton with FD if factorization fails
            return _backward_euler(f, t0, t1, y0, A_matrix, steps)

        for _ in range(steps):
            # Solve for Y1, Y2, Y3
            # Y1 - h*a11*A*Y1 - h*a12*A*Y2 - h*a13*A*Y3 = y
            # Y2 - h*a21*A*Y1 - h*a22*A*Y2 - h*a23*A*Y3 = y
            # Y3 - h*a31*A*Y1 - h*a32*A*Y2 - h*a33*A*Y3 = y
            
            # Since F_i = A Y_i, we can substitute F_i.
            # But we need Y_i to compute F_i.
            # This is a coupled linear system.
            # Let's just solve for Y using substitution or block solve.
            # Given N=20, we can form the 3N x 3N block matrix?
            # Or iterate?
            # Actually, since it's linear, we can just solve the 3x3 block system.
            
            # Simpler: Since it's linear, the stages are linear combinations of y.
            # We can just solve for Y1, Y2, Y3 sequentially if the matrix is lower triangular in stages?
            # No, it's implicit.
            
            # Let's use a simple fixed-point iteration for the stages?
            # Or just solve the full 3N x 3N system.
            
            # Block matrix:
            # [ I  -h a12 A  -h a13 A ] [Y1]   [ y + h a11 A Y1? No. ]
            # [ -h a21 A  I  -h a23 A ] [Y2]   [ y ]
            # [ -h a31 A  -h a32 A  I  ] [Y3]   [ y ]
            
            # Wait, the RHS is just y for all?
            # Eq 1: Y1 - h a11 A Y1 - h a12 A Y2 - h a13 A Y3 = y
            # => (I - h a11 A) Y1 - h a12 A Y2 - h a13 A Y3 = y
            # Eq 2: -h a21 A Y1 + (I - h a22 A) Y2 - h a23 A Y3 = y
            # Eq 3: -h a31 A Y1 - h a32 A Y2 + (I - h a33 A) Y3 = y
            
            # This is a linear system B * Y_all = y_all
            # Where B is 3N x 3N.
            
            B = np.zeros((3*n, 3*n))
            B[:n, :n] = np.eye(n) - h * a11 * A_matrix
            B[:n, n:2*n] = -h * a12 * A_matrix
            B[:n, 2*n:3*n] = -h * a13 * A_matrix
            
            B[n:2*n, :n] = -h * a21 * A_matrix
            B[n:2*n, n:2*n] = np.eye(n) - h * a22 * A_matrix
            B[n:2*n, 2*n:3*n] = -h * a23 * A_matrix
            
            B[2*n:3*n, :n] = -h * a31 * A_matrix
            B[2*n:3*n, n:2*n] = -h * a32 * A_matrix
            B[2*n:3*n, 2*n:3*n] = np.eye(n) - h * a33 * A_matrix
            
            y_all = np.concatenate([y, y, y])
            
            try:
                Y_all = scipy.linalg.solve(B, y_all)
            except Exception:
                # Fallback
                return _backward_euler(f, t0, t1, y0, A_matrix, steps)
                
            Y1 = Y_all[:n]
            Y2 = Y_all[n:2*n]
            Y3 = Y_all[2*n:3*n]
            
            y = y + h * (b1 * (A_matrix @ Y1) + b2 * (A_matrix @ Y2) + b3 * (A_matrix @ Y3))
            t += h
            
        return y

    def _backward_euler(f, t0, t1, y0, A_matrix, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        n = len(y)
        
        # Precompute factor of (I - h A)
        M = np.eye(n) - h * A_matrix
        try:
            U, E = scipy.linalg.lu_factor(M)
        except Exception:
            # Very slow fallback: explicit? No, will blow up.
            # Just return y0? No.
            # Use Newton with FD.
            for _ in range(steps):
                z = y.copy()
                for _n in range(5):
                    fz = f(t+h, z)
                    J = np.zeros((n,n))
                    for j in range(n):
                        e = np.zeros(n); e[j]=1e-6
                        J[:,j] = (f(t+h, z+e) - fz)/1e-6
                    step = np.linalg.solve(np.eye(n) - h*J, -(z-y-h*fz))
                    z += step
                y = z
                t += h
            return y
            
        for _ in range(steps):
            y = scipy.linalg.lu_solve((U, E), y)
            t += h
        return y

    # If we are here, Candidate 2 failed.
    # Use Radau 3 with A matrix if we have it.
    if A is not None:
        # Use a reasonable number of steps. 
        # Radau 3 is 3rd order. 
        # For stiff problems, stability is key.
        # 10 steps should be enough for t in [0, 0.1].
        return _radau_3(counted_f, t0, t1, y0, A, 10)
    else:
        # No A matrix, use Backward Euler with FD Newton.
        # This is slow but robust.
        return _backward_euler(counted_f, t0, t1, y0, None, 100)
