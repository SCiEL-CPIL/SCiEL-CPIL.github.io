"""
Matrix multiplication 2x2 x 2x16 via Strassen decomposition.

The task is to find a rank-minimal exact tensor decomposition for multiplying
a 2x2 matrix A by a 2x16 matrix B.
C = A B

The tensor dimensions are:
- A: 2x2 (4 elements)
- B: 2x16 (32 elements)
- C: 2x16 (32 elements)

The multiplication can be decomposed into 8 independent block multiplications
where each block multiplies A (2x2) by a 2x2 sub-matrix of B.
Since the rank of 2x2 matrix multiplication is 7 (Strassen's algorithm),
and the 8 blocks are independent (disjoint output columns), the total rank
is 8 * 7 = 56.

This is the minimal possible rank for this tensor.
Score = Seed / Rank = 64 / 56 = 1.142857...
"""

import numpy as np

DIMENSIONS = [2, 2, 16]

def _strassen_2x2_2x2():
    """
    Returns the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    U: (4, 7) for A (row-major: a00, a01, a10, a11)
    V: (4, 7) for B (row-major: b00, b01, b10, b11)
    W: (4, 7) for C (row-major: c00, c01, c10, c11)
    
    Note: The standard Strassen formulas are:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b01)
    M7 = (a01 - a11) * (b10 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Indices for A: 0:00, 1:01, 2:10, 3:11
    # Indices for B: 0:00, 1:01, 2:10, 3:11
    # Indices for C: 0:00, 1:01, 2:10, 3:11
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = a00 * (b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = a11 * (b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6 = (a10 - a00) * (b00 + b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7 = (a01 - a11) * (b10 + b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00
    
    return U, V, W

def construct():
    """
    Construct the rank-minimal tensor decomposition for 2x2 * 2x16.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2_2x2()
    
    num_blocks = p // 2  # 8 blocks of 2x2
    rank = 7 * num_blocks  # 56
    
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_blocks):
        col_offset = t * 7
        
        # U is the same for all blocks (A is shared)
        U[:, col_offset:col_offset+7] = U_core
        
        # V mapping:
        # V index is j*p + k.
        # For block t, k ranges from 2*t to 2*t+1.
        # j ranges from 0 to 1.
        # So V rows are:
        # j=0, k=2*t -> index 0*p + 2*t = 2*t
        # j=0, k=2*t+1 -> index 0*p + 2*t+1 = 2*t+1
        # j=1, k=2*t -> index 1*p + 2*t = 16 + 2*t
        # j=1, k=2*t+1 -> index 1*p + 2*t+1 = 16 + 2*t+1
        
        # V_core rows: 0:b00, 1:b01, 2:b10, 3:b11
        # In the block, B_block is:
        # [[B[0, 2t], B[0, 2t+1]],
        #  [B[1, 2t], B[1, 2t+1]]]
        
        V[2*t, col_offset:col_offset+7] = V_core[0]
        V[2*t+1, col_offset:col_offset+7] = V_core[1]
        V[16+2*t, col_offset:col_offset+7] = V_core[2]
        V[16+2*t+1, col_offset:col_offset+7] = V_core[3]
        
        # W mapping:
        # W index is k*n + i.
        # For block t, k ranges from 2*t to 2*t+1.
        # i ranges from 0 to 1.
        # So W rows are:
        # k=2*t, i=0 -> index 2*t*2 + 0 = 4*t
        # k=2*t, i=1 -> index 2*t*2 + 1 = 4*t+1
        # k=2*t+1, i=0 -> index (2*t+1)*2 + 0 = 4*t+2
        # k=2*t+1, i=1 -> index (2*t+1)*2 + 1 = 4*t+3
        
        # W_core rows: 0:C00, 1:C01, 2:C10, 3:C11
        # In the block, C_block is:
        # [[C[0, 2t], C[0, 2t+1]],
        #  [C[1, 2t], C[1, 2t+1]]]
        
        W[4*t, col_offset:col_offset+7] = W_core[0]
        W[4*t+1, col_offset:col_offset+7] = W_core[2]
        W[4*t+2, col_offset:col_offset+7] = W_core[1]
        W[4*t+3, col_offset:col_offset+7] = W_core[3]
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    # Simple self-test
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Random test
    np.random.seed(42)
    A = np.random.randint(-5, 5, (2, 2))
    B = np.random.randint(-5, 5, (2, 16))
    C_expected = A @ B
    
    # Reconstruct C using tensor decomposition
    # C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_reconstructed = np.zeros((2, 16), dtype=np.int64)
    for i in range(2):
        for k in range(16):
            total = 0
            for j in range(2):
                for r in range(56):
                    total += U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
            C_reconstructed[i, k] = total
            
    if np.array_equal(C_expected, C_reconstructed):
        print("Self-test passed.")
    else:
        print("Self-test failed.")
        print(C_expected)
        print(C_reconstructed)
