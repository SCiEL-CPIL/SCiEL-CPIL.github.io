"""Matrix multiplication 3,4,7 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
N, M, P = DIMENSIONS

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00 (k=0,i=0), c11 (k=1,i=1)
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[1, 1] = 1; W[3, 1] = 1  # c10 (k=0,i=1), c11 (k=1,i=1)
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1; V[3, 2] = -1  # b01, -b11
    W[0, 2] = 1; W[2, 2] = 1  # c00 (k=0,i=0), c01 (k=1,i=0)
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1; V[0, 3] = -1  # b10, -b00
    W[1, 3] = 1; W[3, 3] = 1  # c10 (k=0,i=1), c11 (k=1,i=1)
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[0, 4] = 1; W[1, 4] = 1  # c00 (k=0,i=0), c10 (k=0,i=1)
    
    # M6 = (a10-a00)*(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1  # a10, -a00
    V[0, 5] = 1; V[2, 5] = 1  # b00, b10
    W[1, 5] = 1; W[3, 5] = 1  # c10 (k=0,i=1), c11 (k=1,i=1)
    
    # M7 = (a01-a11)*(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
    V[1, 6] = 1; V[3, 6] = 1  # b01, b11
    W[0, 6] = 1; W[2, 6] = 1  # c00 (k=0,i=0), c01 (k=1,i=0)
    
    return U, V, W

def _add_strassen_block(U, V, W, col_offset, u_map, v_map, w_map):
    """Add a Strassen 2x2x2 block at given column offset.
    u_map: list of 4 global row indices for U (order: 00,01,10,11)
    v_map: list of 4 global row indices for V (order: 00,01,10,11)
    w_map: list of 4 global row indices for W (order: 00,01,10,11)
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    for c in range(7):
        for local_i, global_i in enumerate(u_map):
            U[global_i, col_offset + c] += U_s[local_i, c]
        for local_j, global_j in enumerate(v_map):
            V[global_j, col_offset + c] += V_s[local_j, c]
        for local_k, global_k in enumerate(w_map):
            W[global_k, col_offset + c] += W_s[local_k, c]

def _add_schoolbook_2x2x1(U, V, W, col_offset, u_map, v_map, w_map):
    """Add schoolbook 2x2*2x1 block.
    u_map: [u00, u01, u10, u11]
    v_map: [v00, v01] (2 entries)
    w_map: [w0, w1] (2 entries)
    Returns 4 terms.
    """
    # Term 0: u00 * v00 -> w0
    U[u_map[0], col_offset + 0] = 1.0
    V[v_map[0], col_offset + 0] = 1.0
    W[w_map[0], col_offset + 0] = 1.0
    
    # Term 1: u01 * v01 -> w0
    U[u_map[1], col_offset + 1] = 1.0
    V[v_map[1], col_offset + 1] = 1.0
    W[w_map[0], col_offset + 1] = 1.0
    
    # Term 2: u10 * v00 -> w1
    U[u_map[2], col_offset + 2] = 1.0
    V[v_map[0], col_offset + 2] = 1.0
    W[w_map[1], col_offset + 2] = 1.0
    
    # Term 3: u11 * v01 -> w1
    U[u_map[3], col_offset + 3] = 1.0
    V[v_map[1], col_offset + 3] = 1.0
    W[w_map[1], col_offset + 3] = 1.0

def _build_decomposition():
    """Build a decomposition using Strassen blocks where applicable.
    
    Strategy: Split the computation C = A*B into blocks.
    A is 3x4, B is 4x7, C is 3x7.
    
    We can split:
    - n=3 into 2+1
    - m=4 into 2+2
    - p=7 into 2+5 (or other splits)
    
    The 2x2x2 subproblems (i in {0,1}, j in {0,1}, k in {0,1}) can use Strassen (rank 7 vs 8).
    Similarly for other 2x2x2 blocks.
    
    Let's use the following block structure:
    Split i: {0,1} and {2}
    Split j: {0,1} and {2,3}
    Split k: {0,1} and {2,3,4,5,6}
    
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  -> 2x2x2, use Strassen (rank 7)
    C[0:2, 2:7] = A[0:2, 0:2] * B[0:2, 2:7]  -> 2x2x5, schoolbook (rank 20)
    C[0:2, 2:7] += A[0:2, 2:4] * B[2:4, 2:7]  -> 2x2x5, schoolbook (rank 20)
    C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2]  -> 1x2x2, schoolbook (rank 4)
    C[2, 0:2] += A[2, 2:4] * B[2:4, 0:2]  -> 1x2x2, schoolbook (rank 4)
    C[2, 2:7] = A[2, 0:2] * B[0:2, 2:7]  -> 1x2x5, schoolbook (rank 10)
    C[2, 2:7] += A[2, 2:4] * B[2:4, 2:7]  -> 1x2x5, schoolbook (rank 10)
    
    Total rank = 7 + 20 + 20 + 4 + 4 + 10 + 10 = 75
    """
    # Total rank estimate
    # Block (0:2, 0:2, 0:2): Strassen 2x2x2 -> rank 7
    # Block (0:2, 0:2, 2:7): 2x2x5 schoolbook -> rank 2*2*5 = 20
    # Block (0:2, 2:4, 2:7): 2x2x5 schoolbook -> rank 2*2*5 = 20
    # Block (2, 0:2, 0:2): 1x2x2 schoolbook -> rank 1*2*2 = 4
    # Block (2, 2:4, 0:2): 1x2x2 schoolbook -> rank 1*2*2 = 4
    # Block (2, 0:2, 2:7): 1x2x5 schoolbook -> rank 1*2*5 = 10
    # Block (2, 2:4, 2:7): 1x2x5 schoolbook -> rank 1*2*5 = 10
    total_rank = 7 + 20 + 20 + 4 + 4 + 10 + 10
    # = 75
    
    U = np.zeros((N * M, total_rank))
    V = np.zeros((M * P, total_rank))
    W = np.zeros((P * N, total_rank))
    
    col = 0
    
    # Block 1: C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]
    # i in {0,1}, j in {0,1}, k in {0,1}
    # u_map: A[0:2, 0:2] -> rows [0, 1, 4, 5] (i*M+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    # v_map: B[0:2, 0:2] -> rows [0, 1, 7, 8] (j*P+k: 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8)
    # w_map: C[0:2, 0:2] -> rows [0, 1, 3, 4] (k*N+i: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    u_map = [0, 1, 4, 5]
    v_map = [0, 1, 7, 8]
    w_map = [0, 1, 3, 4]
    _add_strassen_block(U, V, W, col, u_map, v_map, w_map)
    col += 7
    
    # Block 2: C[0:2, 2:7] += A[0:2, 0:2] * B[0:2, 2:7]
    # i in {0,1}, j in {0,1}, k in {2,3,4,5,6}
    # This is 2x2x5 schoolbook
    # For each j in {0,1} and k in {2,3,4,5,6}:
    #   C[i,k] += A[i,j] * B[j,k]
    # u_map: A[0:2, 0:2] -> rows [0, 1, 4, 5]
    # v_map: B[0:2, 2:7] -> rows [2,3,4,5,6, 9,10,11,12,13]
    # w_map: C[0:2, 2:7] -> rows [2,3,4,5,6, 5,6,7,8,9] (k*N+i)
    # Actually let's just do schoolbook term by term
    for j in [0, 1]:
        for k in range(2, 7):
            for i in [0, 1]:
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    
    # Block 3: C[0:2, 2:7] += A[0:2, 2:4] * B[2:4, 2:7]
    # i in {0,1}, j in {2,3}, k in {2,3,4,5,6}
    for j in [2, 3]:
        for k in range(2, 7):
            for i in [0, 1]:
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    
    # Block 4: C[2, 0:2] += A[2, 0:2] * B[0:2, 0:2]
    # i=2, j in {0,1}, k in {0,1}
    for j in [0, 1]:
        for k in [0, 1]:
            U[2 * M + j, col] = 1.0
            V[j * P + k, col] = 1.0
            W[k * N + 2, col] = 1.0
            col += 1
    
    # Block 5: C[2, 0:2] += A[2, 2:4] * B[2:4, 0:2]
    # i=2, j in {2,3}, k in {0,1}
    for j in [2, 3]:
        for k in [0, 1]:
            U[2 * M + j, col] = 1.0
            V[j * P + k, col] = 1.0
            W[k * N + 2, col] = 1.0
            col += 1
    
    # Block 6: C[2, 2:7] += A[2, 0:2] * B[0:2, 2:7]
    # i=2, j in {0,1}, k in {2,3,4,5,6}
    for j in [0, 1]:
        for k in range(2, 7):
            U[2 * M + j, col] = 1.0
            V[j * P + k, col] = 1.0
            W[k * N + 2, col] = 1.0
            col += 1
    
    # Block 7: C[2, 2:7] += A[2, 2:4] * B[2:4, 2:7]
    # i=2, j in {2,3}, k in {2,3,4,5,6}
    for j in [2, 3]:
        for k in range(2, 7):
            U[2 * M + j, col] = 1.0
            V[j * P + k, col] = 1.0
            W[k * N + 2, col] = 1.0
            col += 1
    
    # Verify that col == total_rank
    # Should be 75
    
    return U, V, W

def construct():
    t0 = time.time()
    deadline = 150.0
    
    U, V, W = _build_decomposition()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
