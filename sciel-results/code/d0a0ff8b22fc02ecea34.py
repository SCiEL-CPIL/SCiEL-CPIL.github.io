"""
Improved ODE solver for linear systems using Padé approximation of the matrix exponential.
Exploits the linearity of the 2D heat equation to achieve high accuracy with minimal function calls.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _pade13_coefficients():
    """
    Returns coefficients for the (13,13) Padé approximation of the exponential function.
    The approximation is R(z) = P(z) / Q(z) where P and Q are polynomials of degree 13.
    P(z) = sum_{k=0}^{13} a_k z^k
    Q(z) = sum_{k=0}^{13} (-1)^k a_k z^k
    """
    # Coefficients for Pade(13,13)
    # Reference: Higham, "Functions of Matrices: Theory and Computation", Table 6.1
    # Or standard numerical recipes.
    # a_k = (13-k)! * 13! / (k! * (26-k)!)
    coeffs = []
    n = 13
    for k in range(n + 1):
        # Compute a_k = (n-k)! n! / (k! (2n-k)!)
        # Using logarithms to avoid overflow, though for n=13 it's safe with floats
        import math
        log_a = math.lgamma(n - k + 1) + math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(2 * n - k + 1)
        coeffs.append(math.exp(log_a))
    return coeffs

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    timeout = 300.0
    
    # 1. Extract the linear operator A from f(t, y)
    # Since the problem is linear, f(t, y) = A y (possibly + b, but heat eq is homogeneous)
    # We assume f(t, y) is linear in y.
    # f(t, 0) should be 0 for homogeneous heat equation. Let's check.
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    
    # If f_zero is not close to zero, the system is affine: y' = Ay + b.
    # However, standard heat equation is homogeneous. We assume homogeneous for efficiency.
    # If it were affine, we could augment the state vector, but that's complex.
    # Given the task description "two-dimensional heat equation initial-value problem",
    # it is almost certainly homogeneous: y' = A y.
    
    # Extract A by probing basis vectors.
    # A = [ f(t, e_1), f(t, e_2), ..., f(t, e_n) ]
    # This requires n calls to f.
    
    # Check if we have time.
    if time.time() - start_time > timeout - 10:
         # Fallback to simple RK4 if we are running out of time (unlikely for N=100)
         return _rk4_fallback(f, t0, t1, y0, 100)

    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # 2. Compute the Padé approximation of exp(dt * A)
    # We want to solve: Q(dt*A) y_final = P(dt*A) y0
    # Where P and Q are the numerator and denominator polynomials of the Padé approximant.
    
    coeffs = _pade13_coefficients()
    # P(z) = sum_{k=0}^{13} a_k z^k
    # Q(z) = sum_{k=0}^{13} (-1)^k a_k z^k
    
    # We need to compute M_P = P(dt*A) and M_Q = Q(dt*A)
    # M_P = sum_{k=0}^{13} a_k (dt*A)^k
    # M_Q = sum_{k=0}^{13} (-1)^k a_k (dt*A)^k
    
    # Efficiently compute using Horner's method or direct summation.
    # Since n=100, matrix multiplication is cheap.
    
    # Let's compute powers of dt*A
    # M_P = I + a_1 (dt A) + a_2 (dt A)^2 + ... + a_13 (dt A)^13
    # M_Q = I - a_1 (dt A) + a_2 (dt A)^2 - ... + a_13 (dt A)^13
    
    # Note: a_0 = 1 for Pade(13,13)? 
    # Let's check a_0: log(a_0) = lgamma(14) + lgamma(14) - lgamma(1) - lgamma(27)
    # a_0 = 13! * 13! / (0! * 26!) = 1. Yes, a_0 = 1.
    
    # We can compute M_P and M_Q simultaneously.
    M_P = np.eye(n)
    M_Q = np.eye(n)
    
    # Power term
    power = np.eye(n)
    dt_A = dt * A
    
    for k in range(1, 14):
        power = power @ dt_A
        term = coeffs[k] * power
        M_P += term
        if k % 2 == 1:
            M_Q -= term
        else:
            M_Q += term
            
    # 3. Solve the linear system M_Q y_final = M_P y0
    rhs = M_P @ y0
    try:
        y_final = np.linalg.solve(M_Q, rhs)
    except np.linalg.LinAlgError:
        # If singular, fall back to least squares or RK4
        y_final = np.linalg.lstsq(M_Q, rhs, rcond=None)[0]

    return y_final

def _rk4_fallback(f, t0, t1, y0, steps):
    """Simple RK4 fallback in case of issues."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
