"""Matrix multiplication 4x4x7: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each rank-1 term c, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # contributes to C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Term 1: M1 = (a11+a22)*(b11+b22)
    # U: a11 (i=0,j=0) and a22 (i=1,j=1)
    U[0*2+0, 0] = 1  # a11
    U[1*2+1, 0] = 1  # a22
    # V: b11 (j=0,k=0) and b22 (j=1,k=1)
    V[0*2+0, 0] = 1  # b11
    V[1*2+1, 0] = 1  # b22
    # W: contributes to C[0,0] (k=0,i=0) and C[1,1] (k=1,i=1)
    W[0*2+0, 0] = 1  # C[0,0]
    W[1*2+1, 0] = 1  # C[1,1]
    
    # Term 2: M2 = (a21+a22)*b11
    U[1*2+0, 1] = 1  # a21
    U[1*2+1, 1] = 1  # a22
    V[0*2+0, 1] = 1  # b11
    # Contributes to C[1,0] (k=0,i=1) and C[1,1] (k=1,i=1)
    W[0*2+1, 1] = 1  # C[1,0]
    W[1*2+1, 1] = 1  # C[1,1]
    
    # Term 3: M3 = a11*(b12-b22)
    U[0*2+0, 2] = 1  # a11
    V[0*2+1, 2] = 1  # b12
    V[1*2+1, 2] = -1 # -b22
    # Contributes to C[0,1] (k=1,i=0) and C[1,1] (k=1,i=1)
    W[1*2+0, 2] = 1  # C[0,1]
    W[1*2+1, 2] = 1  # C[1,1]
    
    # Term 4: M4 = a22*(b21-b11)
    U[1*2+1, 3] = 1  # a22
    V[1*2+0, 3] = 1  # b21
    V[0*2+0, 3] = -1 # -b11
    # Contributes to C[0,0] (k=0,i=0) and C[1,0] (k=0,i=1)
    W[0*2+0, 3] = 1  # C[0,0]
    W[0*2+1, 3] = 1  # C[1,0]
    
    # Term 5: M5 = (a11+a12)*b22
    U[0*2+0, 4] = 1  # a11
    U[0*2+1, 4] = 1  # a12
    V[1*2+1, 4] = 1  # b22
    # Contributes to C[0,0] (k=0,i=0) and C[1,1] (k=1,i=1)
    W[0*2+0, 4] = 1  # C[0,0]
    W[1*2+1, 4] = 1  # C[1,1]
    
    # Term 6: M6 = (a21-a11)*(b11+b12)
    U[1*2+0, 5] = 1  # a21
    U[0*2+0, 5] = -1 # -a11
    V[0*2+0, 5] = 1  # b11
    V[0*2+1, 5] = 1  # b12
    # Contributes to C[1,1] (k=1,i=1)
    W[1*2+1, 5] = 1  # C[1,1]
    
    # Term 7: M7 = (a12-a22)*(b21+b22)
    U[0*2+1, 6] = 1  # a12
    U[1*2+1, 6] = -1 # -a22
    V[1*2+0, 6] = 1  # b21
    V[1*2+1, 6] = 1  # b22
    # Contributes to C[0,0] (k=0,i=0)
    W[0*2+0, 6] = 1  # C[0,0]
    
    return U, V, W


def _build_block(n, m, p, offset_i, offset_j, offset_k, U, V, W, start_col):
    """Build a block of matrix multiplication.
    For 2x2x2 blocks, uses Strassen. For other sizes, uses schoolbook.
    Returns the number of columns used.
    """
    if n == 2 and m == 2 and p == 2:
        U_block, V_block, W_block = _strassen_2x2x2()
        rank_block = 7
    elif n == 2 and m == 2 and p == 1:
        # 2x2x1: schoolbook rank 4
        rank_block = 4
        U_block = np.zeros((2*2, rank_block))
        V_block = np.zeros((2*1, rank_block))
        W_block = np.zeros((1*2, rank_block))
        col = 0
        for i in range(2):
            for j in range(2):
                U_block[i*2+j, col] = 1
                V_block[j*1+0, col] = 1
                W_block[0*2+i, col] = 1
                col += 1
    else:
        # Schoolbook
        rank_block = n * m * p
        U_block = np.zeros((n*m, rank_block))
        V_block = np.zeros((m*p, rank_block))
        W_block = np.zeros((p*n, rank_block))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_block[i*m+j, col] = 1
                    V_block[j*p+k, col] = 1
                    W_block[k*n+i, col] = 1
                    col += 1
    
    # Place into global matrices with offsets
    U[offset_i*m + offset_j : offset_i*m + offset_j + n*m, start_col:start_col+rank_block] = U_block
    V[offset_j*p + offset_k : offset_j*p + offset_k + m*p, start_col:start_col+rank_block] = V_block
    W[offset_k*n + offset_i : offset_k*n + offset_i + p*n, start_col:start_col+rank_block] = W_block
    
    return rank_block


def construct():
    """Construct rank-minimal exact tensor decomposition for 4x4x7 matrix multiplication.
    
    Strategy: Decompose the 4x4x7 product into blocks.
    - Split the 4x4 A matrix and 4x7 B matrix into 2x2 and 2x1 blocks.
    - Use Strassen (rank 7) for 2x2x2 sub-problems.
    - Use schoolbook (rank 4) for 2x2x1 sub-problems.
    
    The 4x4x7 product can be seen as:
    A (4x4) * B (4x7) = C (4x7)
    
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B into 2x7 blocks: B = [[B0], [B1]] where B0 is 2x7, B1 is 2x7
    Split C into 2x7 blocks: C = [[C0], [C1]] where C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    
    Further split B0, B1 into 2x2 and 2x3 parts? No, let's split by columns of B.
    B is 4x7. Split into B[:, 0:4] (4x4) and B[:, 4:7] (4x3).
    
    C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4 -> 4x4)
    C[:, 4:7] = A * B[:, 4:7]  (4x4 * 4x3 -> 4x3)
    
    For the 4x4x4 part: use 4 Strassen 2x2x2 blocks = 4*7 = 28 rank
    For the 4x4x3 part: split into 4x4x2 + 4x4x1
      4x4x2: 4 Strassen-like? No, 2x2x2 blocks: 4 blocks * 7 = 28
      4x4x1: 4 schoolbook 2x2x1 blocks: 4 blocks * 4 = 16
      Total for 4x4x3: 28 + 16 = 44
    Total: 28 + 44 = 72
    
    Wait, let me reconsider. The known result from the prompt says rank 93.
    Let me use a simpler approach first: just schoolbook with some Strassen optimizations.
    
    Actually, let me just use schoolbook for now to get a valid solution, then optimize.
    Schoolbook rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
