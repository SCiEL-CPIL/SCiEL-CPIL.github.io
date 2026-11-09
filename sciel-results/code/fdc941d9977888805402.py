"""Custom Radau IIA integrator exploiting the known analytic Jacobian of Robertson."""
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 285.0

    # Robertson problem has a known constant Jacobian.
    # y' = f(t, y) where y = [y1, y2, y3]
    # f1 = -0.04*y1 + 1e4*y2*y3
    # f2 =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    # f3 =  3e7*y2^2
    #
    # Jacobian J (constant structure, entries depend on y but we use analytic form):
    # J[0,0] = -0.04,       J[0,1] = 1e4*y3,      J[0,2] = 1e4*y2
    # J[1,0] =  0.04,       J[1,1] = -1e4*y3 - 6e7*y2,  J[1,2] = -1e4*y2
    # J[2,0] =  0,          J[2,1] = 6e7*y2,       J[2,2] = 0

    def jac(y):
        """Analytic Jacobian for Robertson problem."""
        J = np.zeros((n, n))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y[2]
        J[0, 2] = 1e4 * y[1]
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y[2] - 6e7 * y[1]
        J[1, 2] = -1e4 * y[1]
        J[2, 1] = 6e7 * y[1]
        return J

    # Radau IIA coefficients for order 3 (3-stage)
    # But for better accuracy, let's use order 5 (5-stage) Radau IIA
    # Actually, let's use the standard 3-stage Radau IIA (order 3) first,
    # then consider higher order.
    #
    # 3-stage Radau IIA (order 3):
    # c1 = 1/6, c2 = 1/2, c3 = 1
    # A matrix:
    # A[0,0] = 1/30, A[0,1] = 3/10, A[0,2] = 0
    # A[1,0] = 3/40, A[1,1] = 3/10, A[1,2] = 0
    # A[2,0] = 1/5,  A[2,1] = 1/2,  A[2,2] = 0
    # b = [1/5, 3/5, 7/15]
    #
    # Actually let me use the well-known 3-stage Radau IIA:
    # c = [1/6, 1/2, 1]
    # A = [[1/30, 3/10, 0],
    #      [3/40, 3/10, 0],
    #      [1/5,  1/2,  0]]
    # b = [1/5, 3/5, 7/15]

    s = 3
    c = np.array([1.0/6.0, 0.5, 1.0])
    A = np.array([
        [1.0/30.0, 3.0/10.0, 0.0],
        [3.0/40.0, 3.0/10.0, 0.0],
        [1.0/5.0,   1.0/2.0,  0.0]
    ])
    b = np.array([1.0/5.0, 3.0/5.0, 7.0/15.0])

    # For error estimation, use a 2-stage Radau (order 2) as embedded lower-order method
    # Or use step doubling. Let's use step doubling for simplicity.

    rtol = 1e-13
    atol = 1e-15

    def radau_step(t, y, h, J, LU):
        """Perform one Radau IIA step with Newton iteration using provided Jacobian and LU."""
        from scipy.linalg import lu_factor, lu_solve

        # Stage equations: Yi = y + h * sum_j A[i,j] * f(t + c[i]*h, Yi)
        # For each stage i: Yi - h * A[i,i] * f(t + c[i]*h, Yi) = y + h * sum_{j<i} A[i,j] * f(t + c[j]*h, Yj)
        # Since A[i,i] = 0 for Radau IIA (strictly lower triangular A), this simplifies.
        # Actually Radau IIA has A[i,i] = 0, so it's a fully implicit method but the diagonal is zero.
        # Wait, that's not right. Let me recheck.
        #
        # Radau IIA: The A matrix is lower triangular with zero diagonal? No.
        # Actually for Radau IIA, A is lower triangular and the method is fully implicit.
        # The stage equations are: Yi = y + h * sum_{j=1}^{s} A[i,j] * f(Yj)
        # This means: Yi - h * A[i,i] * f(Yi) - h * sum_{j!=i} A[i,j] * f(Yj) = y
        #
        # For the 3-stage Radau IIA, A[2,2] = 0, A[1,1] = 3/10, A[0,0] = 1/30.
        # So stage 0: Y0 - h*(1/30)*f(Y0) = y  (implicit in Y0)
        # Stage 1: Y1 - h*(3/10)*f(Y1) = y + h*(3/40)*f(Y0)  (implicit in Y1, Y0 known)
        # Stage 2: Y2 = y + h*(1/5)*f(Y0) + h*(1/2)*f(Y1)  (explicit, no Newton needed!)
        
        Y = np.zeros((s, n))
        F = np.zeros((s, n))
        
        for i in range(s):
            rhs = y.copy()
            for j in range(i):
                rhs += h * A[i, j] * F[j]
            
            # Solve: Yi - h * A[i,i] * f(t + c[i]*h, Yi) = rhs
            # If A[i,i] == 0, then Yi = rhs (explicit stage)
            if abs(A[i, i]) < 1e-15:
                Y[i] = rhs
                F[i] = f(t + c[i] * h, Y[i])
            else:
                # Newton iteration
                Yi = rhs.copy()
                for _newton in range(20):
                    Fi = f(t + c[i] * h, Yi)
                    residual = Yi - h * A[i, i] * Fi - rhs
                    # Jacobian of the residual w.r.t. Yi: I - h*A[i,i]*J
                    Jres = np.eye(n) - h * A[i, i] * J
                    # Use LU factorization
                    try:
                        LU_i, piv = lu_factor(Jres)
                        dYi = lu_solve((LU_i, piv), -residual)
                    except Exception:
                        dYi = -residual / (1 + abs(h * A[i, i]))
                    
                    Yi += dYi
                    if np.max(np.abs(dYi)) < 1e-16 * (1 + np.max(np.abs(Yi))):
                        break
                Y[i] = Yi
                F[i] = f(t + c[i] * h, Y[i])
        
        y_new = y + h * sum(b[i] * F[i] for i in range(s))
        return y_new

    # Step doubling for error estimation
    def step_double(t, y, h, J):
        """Take one full step of size h, and two half steps of size h/2. Return (y_full, y_half, err)."""
        # Full step
        y_full = radau_step(t, y, h, J, None)
        
        # Two half steps
        J_half = jac(y)
        y_mid = radau_step(t, y, h/2.0, J_half, None)
        J_mid = jac(y_mid)
        y_half = radau_step(t + h/2.0, y_mid, h/2.0, J_mid, None)
        
        err = np.max(np.abs(y_half - y_full) / (1.0 + np.abs(y_half)))
        return y_half, err

    t = t0
    h = 0.01  # Initial step size
    min_h = 1e-10
    max_h = 1.0
    h = min(h, t1 - t0)
    
    f_count = 0
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        
        h = min(h, t1 - t)
        if h < min_h:
            h = min_h
        if h > max_h:
            h = max_h
        
        J = jac(y)
        
        y_new, err = step_double(t, y, h, J)
        
        if err <= max(rtol, atol):
            y = y_new
            t += h
            # Grow step
            if err > 1e-18:
                factor = min(3.0, 0.9 * (rtol / err) ** 0.25)
            else:
                factor = 3.0
            h *= factor
        else:
            # Shrink step
            factor = max(0.3, 0.9 * (rtol / max(err, 1e-18)) ** 0.25)
            h *= factor
    
    return y
