"""Matrix multiplication 6x6x6 with minimal rank decomposition using Strassen."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_UVW():
    """
    Return U2, V2, W2 for 2x2 matrix multiplication via Strassen (rank 7).
    
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    C11 = M1 - M4 + M5 + M7
    C12 = M3 + M5 - M6 - M7
    C21 = M2 + M4
    C22 = M1 - M3 + M6 + M7
    
    Where:
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)
    
    U2: 4 x 7, V2: 4 x 7, W2: 4 x 7
    Row indexing: U2[i*2+j, col], V2[j*2+k, col], W2[k*2+i, col]
    """
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)*(b11+b22)
    # U contributions: a11 (idx 0), a22 (idx 3)
    U2[0, 0] = 1  # a11
    U2[3, 0] = 1  # a22
    # V contributions: b11 (idx 0), b22 (idx 3)
    V2[0, 0] = 1  # b11
    V2[3, 0] = 1  # b22
    # W contributions: C11 gets +M1
    W2[0, 0] = 1  # c11 (k=0,i=0)
    
    # M2 = (a21+a22)*b11
    U2[2, 1] = 1  # a21 (i=1,j=0 -> idx 2)
    U2[3, 1] = 1  # a22 (idx 3)
    V2[0, 1] = 1  # b11
    # W: C21 gets +M2
    W2[2, 1] = 1  # c21 (k=1,i=0 -> idx 2)
    
    # M3 = a11*(b12-b22)
    U2[0, 2] = 1  # a11
    V2[1, 2] = 1  # b12 (j=0,k=1 -> idx 1)
    V2[3, 2] = -1 # b22
    # W: C12 gets +M3, C22 gets -M3
    W2[1, 2] = 1  # c12 (k=1,i=0 -> idx 1)
    W2[3, 2] = -1 # c22 (k=1,i=1 -> idx 3)
    
    # M4 = a22*(b21-b11)
    U2[3, 3] = 1  # a22
    V2[2, 3] = 1  # b21 (j=1,k=0 -> idx 2)
    V2[0, 3] = -1 # b11
    # W: C11 gets -M4, C21 gets +M4
    W2[0, 3] = -1
    W2[2, 3] = 1
    
    # M5 = (a11+a12)*b22
    U2[0, 4] = 1  # a11
    U2[1, 4] = 1  # a12 (i=0,j=1 -> idx 1)
    V2[3, 4] = 1  # b22
    # W: C11 gets +M5, C12 gets +M5
    W2[0, 4] = 1
    W2[1, 4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U2[2, 5] = 1  # a21
    U2[0, 5] = -1 # a11
    V2[0, 5] = 1  # b11
    V2[1, 5] = 1  # b12
    # W: C12 gets -M6, C22 gets +M6
    W2[1, 5] = -1
    W2[3, 5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U2[1, 6] = 1  # a12
    U2[3, 6] = -1 # a22
    V2[2, 6] = 1  # b21
    V2[3, 6] = 1  # b22
    # W: C11 gets +M7, C12 gets -M7, C22 gets +M7
    W2[0, 6] = 1
    W2[1, 6] = -1
    W2[3, 6] = 1
    
    return U2, V2, W2


def construct():
    """
    Construct rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    
    Uses 3x3 block structure with 2x2 blocks, applying Strassen (rank 7) to each.
    Total rank = 27 * 7 = 189.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Get Strassen factors for 2x2
    U2, V2, W2 = _strassen_2x2_UVW()
    
    block_size = 2
    num_blocks = n // block_size  # 3
    
    total_rank = num_blocks * num_blocks * num_blocks * 7  # 27 * 7 = 189
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for I in range(num_blocks):  # row block of A, row block of C
        for J in range(num_blocks):  # col block of A, row block of B
            for K in range(num_blocks):  # col block of B, col block of C
                for r in range(7):
                    # U: A block (I,J), local position (i_loc, j_loc)
                    for i_loc in range(block_size):
                        for j_loc in range(block_size):
                            u_loc = i_loc * block_size + j_loc
                            val = U2[u_loc, r]
                            if val != 0:
                                i = I * block_size + i_loc
                                j = J * block_size + j_loc
                                a_idx = i * m + j
                                U[a_idx, col] = val
                    
                    # V: B block (J,K), local position (j_loc, k_loc)
                    for j_loc in range(block_size):
                        for k_loc in range(block_size):
                            v_loc = j_loc * block_size + k_loc
                            val = V2[v_loc, r]
                            if val != 0:
                                j = J * block_size + j_loc
                                k = K * block_size + k_loc
                                b_idx = j * p + k
                                V[b_idx, col] = val
                    
                    # W: C block (I,K), local position (k_loc, i_loc)
                    # W index is k*n + i
                    for k_loc in range(block_size):
                        for i_loc in range(block_size):
                            w_loc = k_loc * block_size + i_loc
                            val = W2[w_loc, r]
                            if val != 0:
                                k = K * block_size + k_loc
                                i = I * block_size + i_loc
                                c_idx = k * n + i
                                W[c_idx, col] = val
                    
                    col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
