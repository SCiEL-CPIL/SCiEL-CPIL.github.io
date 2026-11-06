import numpy as np
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Solves a 15-dimensional stiff linear ODE system y' = A y + b on [0, 1].
    Assumes the system is autonomous and linear with constant coefficients.
    Extracts A and b using 16 function calls, then computes the exact solution
    using matrix exponentials.
    """
    n = len(y0)
    t0, t1 = t_span
    # The problem statement implies integration on [0, 1], but we use t_span for generality
    # though the extraction assumes constant A, b. If t_span != [0,1], the solution 
    # formula y(t1) = exp(A*(t1-t0)) y0 + integral term holds.
    
    # Extract b = f(0, 0)
    # Note: f might be f(t, y). We assume f(0, 0) gives b.
    zero_vec = np.zeros(n)
    b = np.array(f(t0, zero_vec), dtype=float)
    
    # Extract A
    # f(0, e_i) = A e_i + b => A[:, i] = f(0, e_i) - b
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = np.array(f(t0, e_i), dtype=float) - b
        
    # Compute the solution y(t1) = exp(A * (t1 - t0)) y0 + integral_0^(t1-t0) exp(A * s) b ds
    # Let T = t1 - t0
    T = t1 - t0
    
    # We use the augmented matrix method to compute both exp(A*T) and the integral term simultaneously.
    # Let M = [[A, b], [0, 0]]
    # Then exp(M * T) = [[exp(A*T), int_0^T exp(A*(T-s)) b ds], [0, 1]]
    # Wait, the integral in the augmented block is int_0^T exp(A*(T-s)) b ds?
    # Let's check: d/ds (exp(A(T-s)) b) = -A exp(A(T-s)) b.
    # The standard augmented matrix for y' = Ay + b, z' = b is:
    # [y']   [A b] [y]
    # [z'] = [0 0] [z]
    # Solution: [y(t)] = [exp(At)  int_0^t exp(A(t-s))b ds] [y0]
    #           [z(t)]   [0       1                          ] [z0]
    # If we set z0 = 0, then z(t) = int_0^t exp(A(t-s))b ds.
    # So if we compute expm(T * M), the top-right block is the integral term.
    
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = b
    # M[n, :] is already zeros
    
    # Scale by T
    M_scaled = T * M
    
    E_aug = scipy.linalg.expm(M_scaled)
    
    # y(t1) = E_aug[:n, :n] @ y0 + E_aug[:n, n]
    y_final = E_aug[:n, :n] @ np.array(y0, dtype=float) + E_aug[:n, n]
    
    return y_final
