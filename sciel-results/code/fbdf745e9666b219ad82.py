"""Program for 6x6x6 matrix multiplication tensor decomposition using CW(3x3) ⊗ Strassen(2x2)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 matrices, rank 7
    U2 = [
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, -1, 1],  # A10
        [0, 0, 0, 0, 1, 0, 0],   # A01
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 0],   # B01
        [1, 0, -1, -1, 1, 0, 1], # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, -1], # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],  # C11 = M1 - M2 + M3 + M7
    ]

    # Schoolbook decomposition for 3x3 matrices, rank 27
    # U3 shape: (9, 27), V3 shape: (9, 27), W3 shape: (9, 27)
    U3 = [[0] * 27 for _ in range(9)]
    V3 = [[0] * 27 for _ in range(9)]
    W3 = [[0] * 27 for _ in range(9)]
    
    for r in range(27):
        i_r = r // 9
        j_r = (r // 3) % 3
        k_r = r % 3
        
        # U3 index: i_r * 3 + j_r
        U3[i_r * 3 + j_r][r] = 1
        # V3 index: j_r * 3 + k_r
        V3[j_r * 3 + k_r][r] = 1
        # W3 index: k_r * 3 + i_r
        W3[k_r * 3 + i_r][r] = 1

    # Construct 6x6 decomposition using Kronecker product
    # Rank = 7 * 27 = 189
    rank2 = 7
    rank3 = 27
    total_rank = rank2 * rank3
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Map 6x6 indices to (2x2 block, 3x3 local)
    # i = 2*i2 + i3
    # j = 2*j2 + j3
    # k = 2*k2 + k3
    
    for i2 in range(2):
        for i3 in range(3):
            i = 2 * i2 + i3
            for j2 in range(2):
                for j3 in range(3):
                    j = 2 * j2 + j3
                    for k2 in range(2):
                        for k3 in range(3):
                            k = 2 * k2 + k3
                            
                            # U index: i*m + j = i*6 + j
                            u_idx = i * 6 + j
                            # V index: j*p + k = j*6 + k
                            v_idx = j * 6 + k
                            # W index: k*n + i = k*6 + i
                            w_idx = k * 6 + i
                            
                            for r2 in range(rank2):
                                for r3 in range(rank3):
                                    r = r2 * rank3 + r3
                                    
                                    val_u = U2[i2 * 2 + j2][r2] * U3[i3 * 3 + j3][r3]
                                    if val_u != 0:
                                        U[u_idx][r] += val_u
                                        
                                    val_v = V2[j2 * 2 + k2][r2] * V3[j3 * 3 + k3][r3]
                                    if val_v != 0:
                                        V[v_idx][r] += val_v
                                        
                                    val_w = W2[k2 * 2 + i2][r2] * W3[k3 * 3 + i3][r3]
                                    if val_w != 0:
                                        W[w_idx][r] += val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
