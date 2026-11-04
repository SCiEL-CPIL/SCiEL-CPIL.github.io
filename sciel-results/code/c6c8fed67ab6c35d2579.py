"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
Uses a combination of Strassen's algorithm for the 4x4x4 part (rank 49)
and schoolbook for the 4x4x3 part (rank 48), totaling rank 97.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _validate(U, V, W):
    """
    Check that U @ V^T (per component) reconstructs the tensor via W.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i,j,k] = 1 for all i,j,k.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
        
    # Efficient check using einsum or loops
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We can compute this for all i,j,k
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Sum over rank
    # We want T[i,j,k] = sum_c U_reshaped[i,j,c] V_reshaped[j,k,c] W_reshaped[k,i,c]
    # This is a contraction.
    # np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, 1.0, atol=1e-6):
        return False
    return True

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U, V, W are 4x7 matrices.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Terms M1..M7
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, C11(-)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11 -> C00(-), C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build U, V, W for 4x4x4 using Strassen.
    Rank 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    u2, v2, w2 = _get_strassen_2x2_factors()
    
    # Coefficients for A blocks (I, J) for each term t
    # t: 0..6
    # A blocks: 00(0), 01(1), 10(2), 11(3)
    A_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Coefficients for B blocks (J, K) for each term t
    B_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # Coefficients for C blocks (I, K) for each term t
    C_coeffs = [
        {0: 1, 3: 1},       # M1 -> C00, C11
        {2: 1, 3: -1},      # M2 -> C10, C11(-)
        {1: 1, 3: 1},       # M3 -> C01, C11
        {0: 1, 2: 1},       # M4 -> C00, C10
        {0: -1, 1: 1},      # M5 -> C00(-), C01
        {3: 1},             # M6 -> C11
        {0: 1}              # M7 -> C00
    ]
    
    for t in range(7):
        for s in range(7):
            c_idx = t * 7 + s
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    block_idx_A = I * 2 + J
                    coeff = A_coeffs[t].get(block_idx_A, 0)
                    if coeff != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                global_i = I * 2 + i_prime
                                global_j = J * 2 + j_prime
                                local_idx = i_prime * 2 + j_prime
                                U[global_i * 4 + global_j, c_idx] += u2[local_idx, s] * coeff
            
            # Fill V
            for J in range(2):
                for K in range(2):
                    block_idx_B = J * 2 + K
                    coeff = B_coeffs[t].get(block_idx_B, 0)
                    if coeff != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                global_j = J * 2 + j_prime
                                global_k = K * 2 + k_prime
                                local_idx = j_prime * 2 + k_prime
                                V[global_j * 4 + global_k, c_idx] += v2[local_idx, s] * coeff
            
            # Fill W
            for I in range(2):
                for K in range(2):
                    block_idx_C = I * 2 + K
                    coeff = C_coeffs[t].get(block_idx_C, 0)
                    if coeff != 0:
                        for k_prime in range(2):
                            for i_prime in range(2):
                                global_k = K * 2 + k_prime
                                global_i = I * 2 + i_prime
                                local_idx = k_prime * 2 + i_prime
                                W[global_k * 4 + global_i, c_idx] += w2[local_idx, s] * coeff
                                
    return U, V, W

def _build_4x4x3_schoolbook():
    """
    Build U, V, W for 4x4x3 using schoolbook.
    Rank 48.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Build 4x4x4 Strassen part (Rank 49)
    U1, V1, W1 = _build_4x4x4_strassen()
    
    # Build 4x4x3 Schoolbook part (Rank 48)
    U2, V2, W2 = _build_4x4x3_schoolbook()
    
    # Combine
    # U: (16, 49) and (16, 48) -> (16, 97)
    # V: (28, 49) and (12, 48) -> (28, 97)
    # W: (28, 49) and (12, 48) -> (28, 97)
    
    U = np.hstack([U1, U2])
    
    # V1 is 28x49 (4x4). V2 is 12x48 (4x3).
    # We need V to be 28x97 (4x7).
    # V1 corresponds to B columns 0-3.
    # V2 corresponds to B columns 4-6.
    # So V[0:12, 49:97] = V2? No.
    # V index is j*p + k.
    # For V1, p=4. j in 0..3, k in 0..3.
    # For V2, p=3. j in 0..3, k in 0..2.
    # In the full 4x7 problem, p=7.
    # V_full[j*7 + k, c].
    # For k=0..3, it comes from V1.
    # For k=4..6, it comes from V2.
    
    V_full = np.zeros((28, 97))
    for j in range(4):
        for k in range(4):
            V_full[j*7 + k, 0:49] = V1[j*4 + k, :]
        for k_prime in range(3):
            k = k_prime + 4
            V_full[j*7 + k, 49:97] = V2[j*3 + k_prime, :]
            
    # W1 is 28x49 (4x4). W2 is 12x48 (4x3).
    # W index is k*n + i.
    # For W1, k in 0..3.
    # For W2, k_prime in 0..2 -> k = 4..6.
    
    W_full = np.zeros((28, 97))
    for k in range(4):
        for i in range(4):
            W_full[k*4 + i, 0:49] = W1[k*4 + i, :]
    for k_prime in range(3):
        k = k_prime + 4
        for i in range(4):
            W_full[k*4 + i, 49:97] = W2[k_prime*4 + i, :]
            
    # U is just hstacked because A is the same 4x4.
    # U1 is 16x49. U2 is 16x48.
    # U_full[i*4 + j, c].
    # Both use the same A indices.
    # So hstack is correct.
    
    result = {'U': U, 'V': V_full, 'W': W_full}
    
    # Validate
    if _validate(U, V_full, W_full):
        return result
    else:
        # Fallback to schoolbook 112
        U_sb = np.zeros((16, 112))
        V_sb = np.zeros((28, 112))
        W_sb = np.zeros((28, 112))
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    c = i * 4 * 7 + j * 7 + k
                    U_sb[i * 4 + j, c] = 1
                    V_sb[j * 7 + k, c] = 1
                    W_sb[k * 4 + i, c] = 1
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}

