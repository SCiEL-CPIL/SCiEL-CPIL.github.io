"""Matrix multiplication 2x2 by 2x12: block-diagonal Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 matrix multiplication.
    A (2x2) * B (2x2) -> C (2x2)
    
    Strassen's 7 multiplications:
    m1 = (a00 + a11) * (b00 + b11)
    m2 = (a10 + a11) * b00
    m3 = a00 * (b01 - b11)
    m4 = a11 * (b10 - b00)
    m5 = (a00 + a01) * b11
    m6 = (a10 - a00) * (b00 + b01)
    m7 = (a01 - a11) * (b10 + b11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Indexing:
    U rows: (i,j) -> i*2+j, i in {0,1}, j in {0,1}
    V rows: (j,k) -> j*2+k, j in {0,1}, k in {0,1}
    W rows: (k,i) -> k*2+i, k in {0,1}, i in {0,1}
    
    Tensor entry T[a,b,c] = 1 iff indices are consistent.
    For rank-1 term r: U[a,r] * V[b,r] * W[c,r]
    
    We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    where T[a,b,c] = 1 if a=(i,j), b=(j',k), c=(k',i') with j=j', i=i', k=k'
    """
    # U: 4x7, V: 4x7, W: 4x7
    # Rows of U: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Rows of V: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Rows of W: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # m1 = (a00 + a11) * (b00 + b11)
    # U: a00 and a11 get +1
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    # V: b00 and b11 get +1
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    # W: C00 = m1, C11 = m1
    W[0, 0] = 1  # (k=0,i=0) -> C00
    W[3, 0] = 1  # (k=1,i=1) -> C11
    
    # m2 = (a10 + a11) * b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    # W: C10 = m2
    W[2, 1] = 1  # (k=1,i=0) -> C10
    
    # m3 = a00 * (b01 - b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    # W: C01 = m3, C11 = m3
    W[1, 2] = 1  # (k=1,i=0) -> C01
    W[3, 2] = 1  # (k=1,i=1) -> C11
    
    # m4 = a11 * (b10 - b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    # W: C00 = m4, C10 = m4
    W[0, 3] = 1  # (k=0,i=0) -> C00
    W[2, 3] = 1  # (k=1,i=0) -> C10
    
    # m5 = (a00 + a01) * b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    # W: C00 = -m5, C01 = m5
    W[0, 4] = -1 # (k=0,i=0) -> C00
    W[1, 4] = 1  # (k=1,i=0) -> C01
    
    # m6 = (a10 - a00) * (b00 + b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    # W: C11 = m6
    W[3, 5] = 1  # (k=1,i=1) -> C11
    
    # m7 = (a01 - a11) * (b10 + b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1 # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    # W: C00 = m7
    W[0, 6] = 1  # (k=0,i=0) -> C00
    
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x12 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} where U is (4, rank), V is (24, rank), W is (24, rank).
    
    Strategy: Partition the 2x12 matrix into 6 blocks of 2x2.
    Each block uses Strassen's rank-7 algorithm.
    Total rank = 6 * 7 = 42.
    """
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Get the 2x2 Strassen decomposition
    U_base, V_base, W_base = _strassen_2x2()
    
    # Number of 2x2 blocks in the 2x12 matrix
    num_blocks = p // 2  # 6 blocks
    
    total_rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, total_rank), dtype=float)  # (4, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (24, 42)
    
    for block in range(num_blocks):
        col_offset = block * 7
        
        # U: same for all blocks (the 2x2 matrix A is shared)
        U[:, col_offset:col_offset + 7] = U_base
        
        # V: rows correspond to B[j, k] where j in {0,1}, k in {0,...,11}
        # For block b, the columns are 2*b and 2*b+1
        # V_base rows: (j,k_local) -> j*2 + k_local
        # We need to map to global: row = j * p + (2*b + k_local)
        for j in range(m):
            for k_local in range(2):
                v_local_row = j * 2 + k_local
                v_global_row = j * p + (2 * block + k_local)
                V[v_global_row, col_offset:col_offset + 7] = V_base[v_local_row, :]
        
        # W: rows correspond to C[k, i] where k in {0,...,11}, i in {0,1}
        # Using trace convention: W[k*n + i, r]
        # For block b, the output columns are 2*b and 2*b+1
        # W_base rows: (k_local, i) -> k_local*2 + i
        # We need to map to global: row = (2*b + k_local) * n + i
        for k_local in range(2):
            for i in range(n):
                w_local_row = k_local * 2 + i
                w_global_row = (2 * block + k_local) * n + i
                W[w_global_row, col_offset:col_offset + 7] = W_base[w_local_row, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
