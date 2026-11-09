# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import dct

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Chebyshev Collocation with Picard Iteration
    N = 64  # Number of Chebyshev nodes
    # Chebyshev nodes in [-1, 1], ordered from -1 to 1
    j = np.arange(N)
    x_nodes = np.cos(np.pi * (N - 1 - j) / (N - 1))
    
    # Map to t
    t_nodes = 0.5 * (t1 - t0) * x_nodes + 0.5 * (t1 + t0)
    
    # Initial guess: constant y0
    y_curr = np.tile(y0, (N, 1)).T  # Shape (N, dim)
    
    max_iter = 100
    tol = 1e-15
    
    # Precompute Chebyshev Vandermonde matrix for solving coefficients
    # T[k, j] = T_k(x_nodes[j])
    T_mat = np.polynomial.chebyshev.chebvander(x_nodes, N)
    
    for _ in range(max_iter):
        # Evaluate f at nodes
        f_vals = np.zeros_like(y_curr)
        for i in range(N):
            f_vals[i] = f(t_nodes[i], y_curr[i])
            
        # Solve for Chebyshev coefficients c such that T_mat @ c = f_vals
        # T_mat is (N, N), f_vals is (N, dim)
        try:
            c_coeffs = np.linalg.solve(T_mat, f_vals)
        except np.linalg.LinAlgError:
            c_coeffs = np.linalg.lstsq(T_mat, f_vals, rcond=None)[0]
            
        # Compute new y
        y_new = np.zeros_like(y_curr)
        for dim in range(y0.size):
            c_dim = c_coeffs[:, dim]
            # Integrate
            c_int = np.polynomial.chebyshev.chebint(c_dim, m=1)
            
            # Evaluate antiderivative at x_nodes
            F_vals = np.polynomial.chebyshev.chebval(x_nodes, c_int)
            
            # Adjust constant so F(-1) = 0
            F_at_minus_1 = np.polynomial.chebyshev.chebval(-1, c_int)
            F_vals = F_vals - F_at_minus_1
            
            # Scale by dt/dx
            F_vals = 0.5 * (t1 - t0) * F_vals
            
            y_new[:, dim] = y0[dim] + F_vals
            
        # Check convergence
        if np.max(np.abs(y_new - y_curr)) < tol:
            y_curr = y_new
            break
        y_curr = y_new

    # Return value at t1 (x=1, index -1)
    return y_curr[-1]
# EVOLVE-BLOCK-END
