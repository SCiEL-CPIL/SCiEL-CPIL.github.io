# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank 49 Strassen decomposition for 4x4 matrix multiplication.
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 coefficients
    # A coeffs (2x2) for 7 terms
    A_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    # B coeffs (2x2) for 7 terms
    B_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    # C coeffs (2x2) for 7 terms
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M5
    C_coeffs = [
        [[1, 0], [0, 1]],   # M0: C00, C11
        [[0, 0], [1, -1]],  # M1: C10, -C11
        [[0, 1], [0, 1]],   # M2: C01, C11
        [[1, 0], [1, 0]],   # M3: C00, C10
        [[-1, 1], [0, 0]],  # M4: -C00, C01
        [[0, 0], [0, 1]],   # M5: C11
        [[1, 0], [0, 0]]    # M6: C00
    ]
    
    # Construct U, V, W for 4x4 using tensor product of 2x2 Strassen
    # Index mapping:
    # U[i*4+j, 7*s+t] = A_coeffs[s][S][T] * A_coeffs[t][i2][j2]
    #   i = 2*S + i2, j = 2*T + j2
    # V[j*4+k, 7*s+t] = B_coeffs[s][T][K] * B_coeffs[t][j2][k2]
    #   j = 2*T + j2, k = 2*K + k2
    # W[k*4+i, 7*s+t] = C_coeffs[s][S][K] * C_coeffs[t][i2][k2]
    #   k = 2*K + k2, i = 2*S + i2
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            # U
            for S in range(2):
                for T in range(2):
                    a_outer = A_coeffs[s][S][T]
                    if a_outer == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            a_inner = A_coeffs[t][i2][j2]
                            if a_inner == 0: continue
                            i = 2 * S + i2
                            j = 2 * T + j2
                            row_u = i * 4 + j
                            U[row_u, c_idx] += a_outer * a_inner
                            
            # V
            for T in range(2):
                for K in range(2):
                    b_outer = B_coeffs[s][T][K]
                    if b_outer == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            b_inner = B_coeffs[t][j2][k2]
                            if b_inner == 0: continue
                            j = 2 * T + j2
                            k = 2 * K + k2
                            row_v = j * 4 + k
                            V[row_v, c_idx] += b_outer * b_inner
                            
            # W
            for K in range(2):
                for S in range(2):
                    c_outer = C_coeffs[s][S][K]
                    if c_outer == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c_inner = C_coeffs[t][i2][k2]
                            if c_inner == 0: continue
                            k = 2 * K + k2
                            i = 2 * S + i2
                            row_w = k * 4 + i
                            W[row_w, c_idx] += c_outer * c_inner
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
