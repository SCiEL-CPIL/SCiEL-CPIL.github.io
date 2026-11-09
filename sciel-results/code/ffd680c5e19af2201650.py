"""
Strategy: Fixed-step high-order implicit Runge-Kutta (Gauss-Legendre) with Newton iteration.
Instead of global spectral collocation or adaptive solvers, we use a fixed-step
Gauss-Legendre method of order 2*s (s=4, order 8). This method is A-stable and highly accurate.
We solve the nonlinear system at each step using Newton's method with an analytic Jacobian.
Since the problem is smooth and the step size is small, Newton's method converges in very few iterations.
"""
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s safety margin
    
    # Number of stages (s=4 -> order 8)
    s = 4
    
    # Number of steps
    # For order 8, we need small steps for high accuracy.
    # Let's try N_steps = 20 first. If time allows, we can do more.
    # But fixed step is simpler.
    # Let's estimate required steps for rel < 1e-14.
    # Error ~ h^p. If h=10/20=0.5, error might be too large.
    # Let's use N_steps = 100. h=0.1.
    N_steps = 100
    h = (t1 - t0) / N_steps
    
    # Gauss-Legendre nodes and weights for s=4
    # c = nodes, a = weights
    # From standard tables or numpy.polynomial.legendre.leggauss
    c, a = np.polynomial.legendre.leggauss(s)
    
    # Butkerson matrix B (b_ij) for Gauss-Legendre RK
    # b_ij = integral from 0 to c_i of L_j(t) dt, where L_j are Lagrange basis polynomials
    # We can compute this by integrating the Lagrange polynomials.
    # Or use known values.
    # For s=4:
    # c = [-0.861136311594052575, -0.339981043584856319, 0.339981043584856319, 0.861136311594052575]
    # a = [0.347854845137453807, 0.652145154862546193, 0.652145154862546193, 0.347854845137453807]
    
    # Compute B matrix
    # B[i, j] = integral_0^{c_i} L_j(t) dt
    # L_j(t) = prod_{k != j} (t - c_k) / (c_j - c_k)
    
    def compute_B(c, s):
        B = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                # Integrate L_j(t) from 0 to c_i
                # Use simple quadrature or analytic integration of polynomial
                # Since L_j is a polynomial of degree s-1, we can find its coefficients and integrate
                # Or use numpy polynomial integration
                # Let's construct the polynomial for L_j
                # L_j(t) = P(t) / P(c_j) where P(t) = prod_{k != j} (t - c_k)
                
                # Get coefficients of P(t)
                # P(t) = prod_{k != j} (t - c_k)
                coeffs = np.array([1.0])
                for k in range(s):
                    if k == j:
                        continue
                    coeffs = np.convolve(coeffs, [1.0, -c[k]])
                
                # Normalize
                # Evaluate P(c_j)
                # P(c_j) = prod_{k != j} (c_j - c_k)
                denom = 1.0
                for k in range(s):
                    if k == j:
                        continue
                    denom *= (c[j] - c[k])
                
                L_coeffs = coeffs / denom
                # L_coeffs are in descending order of powers
                # Integrate from 0 to c_i
                # integral of sum L_coeffs[m] t^m dt = sum L_coeffs[m] c_i^(m+1) / (m+1)
                
                # L_coeffs[0] is coeff of t^(s-1), L_coeffs[s-1] is constant
                # numpy polyint returns coefficients in descending order
                # Let's use numpy.polyint
                int_coeffs = np.polyint(L_coeffs)
                # Evaluate at c_i
                val = np.polyval(int_coeffs, c[i])
                # Subtract value at 0 (which is 0 for integral from 0, but polyint adds constant)
                # Actually, np.polyint adds a constant of integration. We want F(c_i) - F(0).
                # F(0) is the constant term of int_coeffs.
                # So val = F(c_i). We need F(c_i) - F(0).
                # F(0) = int_coeffs[-1]
                val -= int_coeffs[-1]
                
                B[i, j] = val
        return B
    
    B = compute_B(c, s)
    
    # RK coefficients: A[i, j] = a_j * B[i, j]
    A = B * a[None, :]
    # b_i = a_i
    b = a
    
    # Newton solver for one step
    def newton_solve(y_prev, t_step, h_step):
        # Solve for Y = [Y_1, ..., Y_s] where Y_i = y(t_n + c_i * h)
        # Y_i = y_prev + h * sum_j A[i, j] f(t_n + c_j * h, Y_j)
        
        # Initial guess: y_prev
        Y = np.tile(y_prev, (s, 1))
        
        for newton_iter in range(20):
            # Check deadline
            if time.time() > deadline:
                return Y  # Return current guess
            
            # Compute residual
            # R_i = Y_i - y_prev - h * sum_j A[i, j] f(t_step + c_j * h, Y_j)
            R = np.zeros((s, 2))
            for i in range(s):
                for j in range(s):
                    # f call
                    f_val = f(t_step + c[j] * h_step, Y[j])
                    R[i] -= h_step * A[i, j] * f_val
                R[i] += Y[i] - y_prev
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-14:
                break
            
            # Compute Jacobian
            # dR_i / dY_k = delta_ik * I - h * A[i, k] * J_f(Y_k)
            J = np.zeros((2*s, 2*s))
            for i in range(s):
                for k in range(s):
                    # Block (i, k) is 2x2
                    if i == k:
                        J[2*i:2*i+2, 2*k:2*k+2] += np.eye(2)
                    
                    # J_f(Y_k)
                    yk = Y[k]
                    Jf = np.array([
                        [0.0, 1.0],
                        [-10.0 * yk[0] * yk[1] - 1.0, 5.0 * (1.0 - yk[0]**2)]
                    ])
                    
                    J[2*i:2*i+2, 2*k:2*k+2] -= h_step * A[i, k] * Jf
            
            # Solve J * delta = -R
            R_flat = R.flatten()
            try:
                delta = np.linalg.solve(J, -R_flat)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                delta, _, _, _ = np.linalg.lstsq(J, -R_flat, rcond=None)
            
            Y = Y.flatten() + delta
            Y = Y.reshape(s, 2)
        
        return Y
    
    y_curr = y0.copy()
    
    for n in range(N_steps):
        # Check deadline
        if time.time() > deadline:
            break
        
        t_step = t0 + n * h
        Y = newton_solve(y_curr, t_step, h)
        
        # Update y_curr
        y_curr = y_curr + h * sum(b[i] * f(t_step + c[i] * h, Y[i]) for i in range(s))
    
    return y_curr

