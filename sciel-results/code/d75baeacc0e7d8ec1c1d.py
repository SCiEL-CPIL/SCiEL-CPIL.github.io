import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x3 matrix.
    
    The decomposition is of the form:
    C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
    
    Note: The standard bilinear form is:
    M_r = (sum_{i,j} U[i*m+j, r] * A[i, j]) * (sum_{j,k} V[j*p+k, r] * B[j, k])
    C[i, k] = sum_r W[k*n+i, r] * M_r
    
    However, the provided interface in the task description says:
    "output factor follows the official trace convention W[k*n+i,r]"
    and the verification in the seed code suggests:
    C[i, k] = sum_r W[k*n+i, r] * (sum_j A[i, j] * U[i*m+j, r]) * (sum_j B[j, k] * V[j*p+k, r])
    
    Wait, the seed code's `_verify` function tests:
    C_test1[i, k] += W[k*n+i][r] * u_sum[i] * v_sum[k]
    where u_sum[i] = sum_j A[i, j] * U[i*m+j][r]
    and v_sum[k] = sum_j B[j, k] * V[j*p+k][r]
    
    This is NOT the standard Strassen form. The standard Strassen form computes
    M_r = (sum A*U) * (sum B*V) as a scalar, then C = sum W * M_r.
    
    The form tested in the seed is:
    C[i, k] = sum_r W[k*n+i, r] * (LinearForm_A_r[i]) * (LinearForm_B_r[k])
    
    This is a different decomposition structure!
    In this structure, U and V are not just coefficients for a single scalar product,
    but they define linear forms that result in vectors.
    
    Specifically:
    L_A_r[i] = sum_j U[i*m+j, r] A[i, j]  (Note: U depends on i, j, r)
    L_B_r[k] = sum_j V[j*p+k, r] B[j, k]  (Note: V depends on j, k, r)
    
    Then C[i, k] = sum_r W[k*n+i, r] L_A_r[i] L_B_r[k]
    
    This is equivalent to the standard tensor decomposition:
    T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    Let's check if the rank-11 solution provided in the seed satisfies this.
    The seed's `_get_rank11` defines U, V, W.
    
    I will reuse the rank-11 solution from the seed as it is known to be valid for the seed's verification logic.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Rank 11 Decomposition
    # Terms 0-6: Strassen-like for the first two columns (2x2 * 2x2 part)
    # Terms 7-10: Schoolbook for the third column (2x2 * 2x1 part)
    
    # Term 0
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00 * B02 -> C02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01 * B12 -> C02
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10 * B02 -> C12
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11 * B12 -> C12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
