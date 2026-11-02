"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact for 2x2x8 multiplication."""
    n, m, p = 2, 2, 8
    rank = len(U[0])
    # Build tensor T[i, j, k] which should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication (A is n x m, B is m x p, C is n x p)
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T[i, j, k] = delta (i.e., T[i,j,k] = 1 if this term contributes to C[i,k])
    # Actually, the multiplication tensor is: for each (i, j, k), the coefficient is 1
    # since C[i,k] += A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i, j, k
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i, j, k
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if s != 1:
                    return False
    return True

def _verify_numeric(U, V, W):
    """Faster numeric verification using float64."""
    n, m, p = 2, 2, 8
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    U: shape (4, 7) for A entries (i, j)
    V: shape (4, 7) for B entries (j, k)
    W: shape (4, 7) for C entries (k, i) with convention W[k*n+i, c]
    """
    n = 2
    # Strassen's 7 products for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 - m4 + m5 + m7
    # c12 = m3 + m4
    # c21 = m2 + m5
    # c22 = m1 - m2 + m3 + m6
    
    # Index convention: U[i*m+j, c] where i, j in {0,1}
    # V[j*p+k, c] where j, k in {0,1}
    # W[k*n+i, c] where k, i in {0,1}
    
    # U rows: (0,0)=a11, (0,1)=a12, (1,0)=a21, (1,1)=a22
    # V rows: (0,0)=b11, (0,1)=b12, (1,0)=b21, (1,1)=b22
    # W rows: (0,0)=c11 (k=0,i=0), (0,1)=c12 (k=1,i=0), (1,0)=c21 (k=0,i=1), (1,1)=c22 (k=1,i=1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1  # a11, a22
    V[0, 0] = 1; V[3, 0] = 1  # b11, b22
    W[0, 0] = 1; W[3, 0] = 1  # c11, c22
    
    # m2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1  # a21, a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1; W[3, 1] = -1  # c21, -c22
    
    # m3 = a11(b12-b22)
    U[0, 2] = 1  # a11
    V[1, 2] = 1; V[3, 2] = -1  # b12, -b22
    W[1, 2] = 1; W[3, 2] = 1  # c12, c22
    
    # m4 = a22(b21-b11)
    U[3, 3] = 1  # a22
    V[2, 3] = 1; V[0, 3] = -1  # b21, -b11
    W[0, 3] = -1; W[1, 3] = 1  # -c11, c12
    
    # m5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1  # a11, a12
    V[3, 4] = 1  # b22
    W[0, 4] = 1; W[2, 4] = 1  # c11, c21
    
    # m6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1  # a21, -a11
    V[0, 5] = 1; V[1, 5] = 1  # b11, b12
    W[3, 5] = 1  # c22
    
    # m7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1  # a12, -a22
    V[2, 6] = 1; V[3, 6] = 1  # b21, b22
    W[0, 6] = 1  # c11
    
    return U, V, W

def _block_decompose_2x2x8():
    """
    Use block structure: 2x2 * 2x8 = 2x8.
    View A as 2x2, B as 2x8 (two 2x4 blocks or four 2x2 blocks).
    
    Strategy: Split B into four 2x2 blocks: B = [B1 B2 B3 B4] where each Bj is 2x2.
    Then C = A * B = [A*B1, A*B2, A*B3, A*B4].
    Each A*Bj is a 2x2x2 multiplication.
    
    Using Strassen for each block independently gives rank 7*4 = 28.
    But we can share structure.
    
    Actually, let's think more carefully. We have 4 blocks of 2x2x2.
    The Strassen decomposition for each uses the same A but different B blocks.
    
    For block j (j=0..3), B block is columns 2j, 2j+1 of B.
    V entries for block j: V[j*p + k, c] where j is the row of B (0 or 1), k is the column within the block (0 or 1).
    So V row index = j*8 + k for the full 2x8 matrix.
    
    For each of the 4 blocks, we use the same U (since A is the same) but different V columns.
    W for block j: W[k*n+i, c] where k is the column within block j, i is the row of C.
    So W row index = k*2 + i, but k here is the global column index... 
    
    Wait, let me re-read the convention: W[k*n+i, r] where k is the column index of C (0..p-1=0..7) and i is the row index (0..n-1=0..1).
    So W has p*n = 8*2 = 16 rows.
    
    For block j, the output columns are 2j and 2j+1.
    W[(2j)*n + i, c] and W[(2j+1)*n + i, c] for i=0,1.
    
    So for each block j, we have a 2x2x2 Strassen decomposition with:
    - U: same 4x7 matrix
    - V: 4x7 matrix where row (j_B, k_local) maps to global row j_B*8 + (2j + k_local)
    - W: 4x7 matrix where row (k_local, i) maps to global row (2j + k_local)*2 + i
    
    Total rank = 4 * 7 = 28.
    """
    n, m, p = 2, 2, 8
    U_base, V_base, W_base = _strassen_2x2x2()
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(4):
        offset = block * 7
        # U is the same for all blocks
        U[:, offset:offset+7] = U_base
        
        # V: for block, B rows are 0,1; columns are 2*block, 2*block+1
        # V_base row (j, k_local) -> global row j * p + (2*block + k_local)
        for j in range(2):
            for k_local in range(2):
                V[j * p + (2 * block + k_local), offset:offset+7] = V_base[j, k_local]
        
        # W: for block, C columns are 2*block, 2*block+1; rows are 0,1
        # W_base row (k_local, i) -> global row (2*block + k_local) * n + i
        for k_local in range(2):
            for i in range(2):
                W[(2 * block + k_local) * n + i, offset:offset+7] = W_base[k_local, i]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try the Strassen-based 28-rank construction
    U, V, W = _block_decompose_2x2x8()
    
    # Verify
    if _verify_numeric(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
