"""Matrix multiplication 2x4x4 tensor decomposition using Strassen blocks."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    
    # We use the 4-block Strassen decomposition
    rank = 4 * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 7 terms for 2x2 block multiplication
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {local_a_idx: coeff} where local_a_idx encodes (row, col) in 2x2
    # v_dict: {local_b_idx: coeff}
    # w_dict: {local_c_idx: coeff}
    
    # Local index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        # 0: M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 1: M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # 2: M3 = a00(b01-b11) -> c01 += M3, c11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # 3: M4 = a11(b10-b00) -> c00 += M4, c10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # 4: M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # 5: M6 = (a10-a00)(b00+b01) -> c11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 6: M7 = (a01-a11)(b10+b11) -> c00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Define 4 blocks for 2x4 * 4x4:
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    
    # For each block b:
    # j_offset = 2 * (b // 2)  # column offset in A, row offset in B
    # k_offset = 2 * (b % 2)   # column offset in B, column offset in C
    
    for b in range(4):
        j_off = 2 * (b // 2)
        k_off = 2 * (b % 2)
        
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = terms[s]
            
            # U: A[i, j] where i in {0,1}, j in {j_off, j_off+1}
            # Local x_idx: 0->(i=0,j=j_off), 1->(i=0,j=j_off+1), 
            #              2->(i=1,j=j_off), 3->(i=1,j=j_off+1)
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                j = j_off + j_local
                a_idx = i * m + j  # i*4 + j
                U[a_idx][col] += coeff
            
            # V: B[j, k] where j in {j_off, j_off+1}, k in {k_off, k_off+1}
            # Local y_idx: 0->(j=j_off,k=k_off), 1->(j=j_off,k=k_off+1), 
            #              2->(j=j_off+1,k=k_off), 3->(j=j_off+1,k=k_off+1)
            for y_idx, coeff in v_dict.items():
                j_local = y_idx // 2
                k_local = y_idx % 2
                j = j_off + j_local
                k = k_off + k_local
                b_idx = j * p + k  # j*4 + k
                V[b_idx][col] += coeff
            
            # W: C[i, k] where i in {0,1}, k in {k_off, k_off+1}
            # Local z_idx: 0->(i=0,k=k_off), 1->(i=0,k=k_off+1), 
            #              2->(i=1,k=k_off), 3->(i=1,k=k_off+1)
            # Trace convention: W[k*n+i, r]
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                k = k_off + k_local
                c_idx = k * n + i  # k*2 + i
                W[c_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
