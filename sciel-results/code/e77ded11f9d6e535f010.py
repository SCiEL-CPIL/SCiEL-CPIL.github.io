import numpy as np
from scipy.linalg import eigh
import time

def construct():
    # Hermite polynomial values at 0 (physicists' convention)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Normalization constants N_n = 2^n * n! (sqrt(pi) cancels)
    # N_0 = 1
    # N_4 = 16 * 24 = 384
    # N_8 = 256 * 40320 = 10321920
    # N_12 = 4096 * 479001600 = 1962007275520
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1962007275520.0
    
    # Energy / Moment coefficients E_n = (n + 0.5) * N_n
    # This represents the numerator of <x^2> for the diagonal term
    E0 = 0.5 * N0
    E4 = 4.5 * N4
    E8 = 8.5 * N8
    E12 = 12.5 * N12
    
    # The constraint g(0) = 0 implies:
    # c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    # We want to minimize R = <x^2> = (sum c_i^2 E_i) / (sum c_i^2 N_i)
    # Subject to c12 = -v . c_free where c_free = [c0, c4, c8]
    
    # Let u = [c0, c4, c8].
    # c12 = - (H0_0/H12_0)*u0 - (H4_0/H12_0)*u1 - (H8_0/H12_0)*u2
    v = np.array([H0_0, H4_0, H8_0]) / H12_0
    
    # The total Norm N(u) = u0^2 N0 + u1^2 N4 + u2^2 N8 + c12^2 N12
    # c12^2 = (v . u)^2 = u^T (v v^T) u
    # So N(u) = u^T (diag(N0, N4, N8) + N12 * v v^T) u
    
    # The total Energy E(u) = u0^2 E0 + u1^2 E4 + u2^2 E8 + c12^2 E12
    # E(u) = u^T (diag(E0, E4, E8) + E12 * v v^T) u
    
    M_N = np.diag([N0, N4, N8]) + N12 * np.outer(v, v)
    M_E = np.diag([E0, E4, E8]) + E12 * np.outer(v, v)
    
    # Solve generalized eigenvalue problem M_E x = lambda M_N x
    # We want the smallest lambda.
    try:
        eigenvalues, eigenvectors = eigh(M_E, M_N)
        # eigh returns eigenvalues in ascending order
        min_lambda = eigenvalues[0]
        optimal_u = eigenvectors[:, 0]
    except Exception as e:
        # Fallback if solver fails (should not happen for 3x3 SPD)
        # Use the previous heuristic or a simple guess
        # The vector [1, -1/12, 0] makes c12=0, which is a valid point
        # But we want the minimum.
        # Let's just return a known good-ish point if it fails
        optimal_u = np.array([1.0, -1.0/12.0, 0.0])
        
    c0, c4, c8 = optimal_u
    # c12 is fixed by constraint, but we only return the 3 free coefficients
    # The function signature returns "three coefficients ... (degrees 0, 4, 8)"
    
    return np.array([c0, c4, c8])
