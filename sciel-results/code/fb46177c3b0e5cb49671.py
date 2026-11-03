"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
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
    
    # U rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # V rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # W rows: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3 (k*2+i)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22): U[0,0]=1, U[3,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[3,0]=-1, W[1,0]=0, W[2,0]=0
    # Wait, W[k*2+i, c]: C11 gets M1, so W[0*2+0, 0] = 1
    # C22 gets M1, so W[1*2+1, 0] = 1 (k=1, i=1 -> index 3)
    # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    
    # Let me be more careful:
    # W[k*2+i, c] contributes to C[i,k]
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # W indices: k*2+i
    # C[0,0]: i=0,k=0 -> W[0, c]
    # C[0,1]: i=0,k=1 -> W[2, c]
    # C[1,0]: i=1,k=0 -> W[1, c]
    # C[1,1]: i=1,k=1 -> W[3, c]
    
    # M1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=1, W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: U[2,1]=1, U[3,1]=1; V[0,1]=1; W[1,1]=1, W[3,1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: U[0,2]=1; V[1,2]=1, V[3,2]=-1; W[2,2]=1, W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: U[3,3]=1; V[2,3]=1, V[0,3]=-1; W[0,3]=1, W[1,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: U[0,4]=1, U[1,4]=1; V[3,4]=1; W[0,4]=-1, W[2,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[1,5]=1; W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: U[1,6]=1, U[3,6]=-1; V[2,6]=1, V[3,6]=1; W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    The 6x6 matrix is divided into a 3x3 grid of 2x2 blocks.
    Matrix multiplication C = A*B where A, B, C are 6x6.
    In block form: C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    where each block is 2x2.
    
    For each of the 27 block multiplications (i,j,k in {0,1,2}), we use Strassen (rank 7).
    Total rank = 27 * 7 = 189.
    
    U: (36, 189), V: (36, 189), W: (36, 189)
    U row index: (block_i, local_i, block_j, local_j) -> (block_i*2+local_i)*6 + (block_j*2+local_j)
    Actually, U[i*m+j, c] where i,j are 0..5.
    i = block_i * 2 + local_i, j = block_j * 2 + local_j
    U row = i*6 + j = (block_i*2+local_i)*6 + (block_j*2+local_j)
    
    V[j*p+k, c]: j = block_j*2+local_j, k = block_k*2+local_k
    V row = j*6 + k = (block_j*2+local_j)*6 + (block_k*2+local_k)
    
    W[k*n+i, c]: k = block_k*2+local_k, i = block_i*2+local_i
    W row = k*6 + i = (block_k*2+local_k)*6 + (block_i*2+local_i)
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2()
    
    # For each block (bi, bj, bk) in {0,1,2}^3, and each Strassen term c in {0..6}
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    # U: for each local_i in {0,1}, local_j in {0,1}
                    # U row = (bi*2+li)*6 + (bj*2+lj)
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    # V: for each local_j in {0,1}, local_k in {0,1}
                    # V row = (bj*2+lj)*6 + (bk*2+lk)
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    # W: for each local_k in {0,1}, local_i in {0,1}
                    # W row = (bk*2+lk)*6 + (bi*2+li)
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 189)
    best = _block_decomposition_6x6x6()
    best_rank = 189
    
    # Verify it
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
