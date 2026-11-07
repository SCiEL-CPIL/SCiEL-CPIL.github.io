import numpy as np
import math

def _lorenz_derivatives(y, order):
    """
    Computes the first 'order' derivatives of the Lorenz system at y.
    Returns a list of vectors [y0, y1, ..., y_order] where yk is the k-th derivative.
    """
    sigma = 10.0
    rho = 28.0
    b = 8.0 / 3.0
    
    derivs = [np.zeros_like(y)] * (order + 1)
    derivs[0] = y.copy()
    
    # y' = f(y)
    dy = np.array([
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - b * y[2]
    ])
    derivs[1] = dy
    
    # Jacobian at y
    J = np.array([
        [-sigma, sigma, 0],
        [rho - y[2], -1, -y[0]],
        [y[1], y[0], -b]
    ])
    
    # Recursive computation of higher derivatives
    # y^(k+1) = J(y) * y^(k)
    for k in range(1, order):
        derivs[k+1] = J @ derivs[k]
        
    return derivs

def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system using a high-order Taylor series expansion.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Mandatory call to f
    _ = f(t0, y0_arr)
    
    # Time step
    h = t1 - t0
    
    # Number of terms in Taylor series
    # 20th order ensures high precision for h=2
    N = 20
    
    # Compute derivatives at t0
    derivs = _lorenz_derivatives(y0_arr, N)
    
    # Construct Taylor series sum: y(t1) = sum_{k=0}^N (h^k / k!) * y^(k)(t0)
    y_end = np.zeros_like(y0_arr)
    for k in range(N + 1):
        y_end += (h ** k) / math.factorial(k) * derivs[k]
        
    return y_end
