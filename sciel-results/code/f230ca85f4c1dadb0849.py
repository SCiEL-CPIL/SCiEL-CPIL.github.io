"""
ODE solver: 15-dimensional stiff linear system.

Strategy:
- Call f once at (0, y0) to get the initial derivative.
- Use a Taylor series expansion of the state:
    y(t) = sum_{k=0}^{K} (t^k / k!) * y^{(k)}(0)
  where y^{(k+1)}(0) = A^T * y^{(k)}(0) for a linear system y' = A y + b.
- For a linear system, we can extract A and b from a few evaluations,
  but to keep f-call count minimal we instead use the fact that for
  a linear autonomous system y' = A*y + b, the derivatives at t=0 can
  be generated recursively IF we can compute A*y + b. Since we only
  have f, we can use the following trick:
  
  For a linear system, f(t, y) = A*y + b (assuming time-independent).
  We can estimate A and b by calling f at y0 and a perturbed y.
  But the directive says to use minimal 16 calls to identify A and b.
  
  Actually, a simpler and more robust approach for a 15-dim system:
  Use the fact that if the system is linear and time-independent, we can
  use the Krylov subspace or simply compute the Taylor coefficients by
  repeated application of the linear operator. Since we can't get A
  directly without many calls, let's use a different strategy:
  
  Use a high-order Runge-Kutta or Taylor method with automatic step
  size control, leveraging the fact that for stiff linear systems,
  implicit methods or matrix exponential methods are ideal.
  
  The directive specifically mentions Taylor series of the matrix
  exponential. So let's extract A and b first using 16 calls (1 for
  f(0,y0), and 15 more for column vectors of A), then compute
  y(1) = expm(A)*y0 + A^{-1}*(expm(A) - I)*b.
"""

import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1]
    with initial condition y0.
    
    Assumes a 15-dimensional linear time-invariant system.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract the linear operator A and offset b from f.
    # For a linear time-invariant system: f(t, y) = A @ y + b
    # We call f at y0 to get b + A@y0.
    # Then we call f at y0 + e_i * eps to get b + A@(y0 + eps*e_i)
    # so (f(y0 + eps*e_i) - f(y0)) / eps gives column i of A.
    
    y0_orig = y0.copy()
    f0 = np.array(f(t0, y0), dtype=float)
    
    # Estimate A using central differences or forward differences
    # Using forward differences with a small step
    eps = 1e-8
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_pert = y0_orig.copy()
        y_pert[j] += eps
        f_pert = np.array(f(t0, y_pert), dtype=float)
        A[:, j] = (f_pert - f0) / eps
    
    # b = f0 - A @ y0
    b = f0 - A @ y0_orig
    
    # Now solve y' = A*y + b analytically using matrix exponential.
    # y(t) = exp(A*t) * y0 + int_0^t exp(A*(t-s)) * b ds
    #      = exp(A*t) * y0 + A^{-1} * (exp(A*t) - I) * b  if A is invertible
    # More generally: y(t) = exp(A*t) * y0 + int_0^t exp(A*(t-s)) * b ds
    # The integral can be computed as: A^{-1} * (exp(A*t) - I) * b
    # But if A is singular, we can use the series expansion.
    
    # Use scipy.linalg.expm for the matrix exponential
    from scipy.linalg import expm
    
    h = t1 - t0
    expAh = expm(A * h)
    
    # Compute the integral term: int_0^h exp(A*s) ds * b
    # This equals A^{-1} * (exp(A*h) - I) * b if A is invertible.
    # For robustness, use the series expansion or scipy's phmmexp.
    # Let's use the series expansion for the integral:
    # int_0^h exp(A*s) ds = sum_{k=0}^{K} A^k * h^{k+1} / (k+1)!
    
    # Or simply use the identity: if A is invertible, 
    # integral = A^{-1} * (exp(A*h) - I)
    # For a 15x15 matrix, this should be fine.
    
    try:
        # Check if A is invertible
        det_A = np.linalg.det(A)
        if abs(det_A) > 1e-12:
            integral_term = np.linalg.solve(A, (expAh - np.eye(n)) @ b)
        else:
            # Use series expansion
            integral_term = np.zeros(n, dtype=float)
            term = b.copy()
            A_pow = np.eye(n, dtype=float)
            for k in range(30):
                A_pow = A_pow @ A
                term = term @ A * (h / (k + 2))
                integral_term += term
                if np.max(np.abs(term)) < 1e-16 * np.max(np.abs(integral_term)) + 1e-300:
                    break
    except np.linalg.LinAlgError:
        # Use series expansion as fallback
        integral_term = np.zeros(n, dtype=float)
        term = b.copy()
        A_pow = np.eye(n, dtype=float)
        for k in range(30):
            A_pow = A_pow @ A
            term = term @ A * (h / (k + 2))
            integral_term += term
            if np.max(np.abs(term)) < 1e-16 * np.max(np.abs(integral_term)) + 1e-300:
                break
    
    y_final = expAh @ y0_orig + integral_term
    return y_final
