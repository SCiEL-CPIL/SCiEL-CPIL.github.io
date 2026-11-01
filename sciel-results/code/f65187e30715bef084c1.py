"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit

    n, m, p = DIMENSIONS
    # Schoolbook decomposition (rank 28) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen-based decomposition for rank 25
    # View B as four 2x2 blocks (first 6 cols) + one 2x1 block (last col)
    # Actually, let's be more careful.
    # B is 2x7. Split into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1).
    # A is 2x2.
    # C = A*B = A*B1 + A*B2 + A*B3 + A*B4
    # Use Strassen for A*B1, A*B2, A*B3: 7 terms each = 21 terms.
    # Use schoolbook for A*B4: 4 terms (since B4 is 2x1, product is 2x1, but rank contribution is 2*1*1=2? No, A is 2x2, B4 is 2x1, result is 2x1. Rank = 2*2*1 = 4).
    # Total rank = 21 + 4 = 25.

    # Let's construct this.
    # Indices: A[i,j], B[j,k], C[i,k]
    # U has shape (4, rank), V has shape (14, rank), W has shape (14, rank)

    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(u_row_offset, v_col_start, w_col_start, rank_offset):
        # A is 2x2: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B_block is 2x2: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # Strassen terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C[0,0] = M1 - M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 - M3 + M6

        # U rows: 0=a, 1=b, 2=c, 3=d
        # V rows: v_col_start=0, 1=f, 2=g, 3=h (for the block)
        # W rows: w_col_start=0, 1=1, 2=2, 3=3 (for the block)

        # Term 1: M1 = (a+d)(e+h)
        idx = rank_offset
        U[0][idx] = 1; U[3][idx] = 1
        V[v_col_start][idx] = 1; V[v_col_start + 3][idx] = 1
        W[w_col_start][idx] = 1
        W[w_col_start + 1][idx] = 1
        W[w_col_start + 2][idx] = 1
        W[w_col_start + 3][idx] = 1

        # Term 2: M2 = (c+d)e
        idx = rank_offset + 1
        U[2][idx] = 1; U[3][idx] = 1
        V[v_col_start][idx] = 1
        W[w_col_start + 1][idx] = -1
        W[w_col_start + 2][idx] = 1
        W[w_col_start + 3][idx] = -1

        # Term 3: M3 = a(f-h)
        idx = rank_offset + 2
        U[0][idx] = 1
        V[v_col_start + 1][idx] = 1
        V[v_col_start + 3][idx] = -1
        W[w_col_start + 1][idx] = 1
        W[w_col_start + 3][idx] = -1

        # Term 4: M4 = d(g-e)
        idx = rank_offset + 3
        U[3][idx] = 1
        V[v_col_start + 2][idx] = 1
        V[v_col_start][idx] = -1
        W[w_col_start + 2][idx] = 1
        W[w_col_start + 3][idx] = -1

        # Term 5: M5 = (a+b)h
        idx = rank_offset + 4
        U[0][idx] = 1; U[1][idx] = 1
        V[v_col_start + 3][idx] = 1
        W[w_col_start][idx] = -1
        W[w_col_start + 1][idx] = 1

        # Term 6: M6 = (c-a)(e+f)
        idx = rank_offset + 5
        U[2][idx] = 1; U[0][idx] = -1
        V[v_col_start][idx] = 1
        V[v_col_start + 1][idx] = 1
        W[w_col_start][idx] = -1
        W[w_col_start + 3][idx] = 1

        # Term 7: M7 = (b-d)(g+h)
        idx = rank_offset + 6
        U[1][idx] = 1; U[3][idx] = -1
        V[v_col_start + 2][idx] = 1
        V[v_col_start + 3][idx] = 1
        W[w_col_start][idx] = 1
        W[w_col_start + 2][idx] = 1

    # Block 1: B[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0)
    # Block 2: B[0:2, 2:4]
    add_strassen_block(0, 2, 2, 7)
    # Block 3: B[0:2, 4:6]
    add_strassen_block(0, 4, 4, 14)

    # Block 4: B[0:2, 6:7] (2x1)
    # Schoolbook:
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    idx = 21
    U[0][idx] = 1
    V[6][idx] = 1
    W[6][idx] = 1

    idx = 22
    U[1][idx] = 1
    V[7][idx] = 1
    W[6][idx] = 1

    idx = 23
    U[2][idx] = 1
    V[6][idx] = 1
    W[13][idx] = 1

    idx = 24
    U[3][idx] = 1
    V[7][idx] = 1
    W[13][idx] = 1

    # Verify correctness
    # Build tensor T from U,V,W
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Matrix mult tensor: T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # In our convention: C[i,k] = sum_j A[i,j] B[j,k]
    # So tensor element is 1 for all (i,j,k).
    
    # Let's verify with a random matrix
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6, 7, 8, 9, 10, 11],
                  [12, 13, 14, 15, 16, 17, 18]])
    C_expected = A @ B
    
    C_computed = np.zeros((2, 7))
    for c in range(rank):
        u_col = np.array([U[i*m+j][c] for i in range(n) for j in range(m)])
        v_col = np.array([V[j*p+k][c] for j in range(m) for k in range(p)])
        w_col = np.array([W[k*n+i][c] for k in range(p) for i in range(n)])
        
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        
        # C += u_mat @ v_mat * w_mat? No.
        # The decomposition is: T = sum_c u_c outer v_c outer w_c
        # C[i,k] = sum_j sum_c A[i,j] B[j,k] W[k,i,c] U[i,j,c] V[j,k,c]
        # This is not a simple matmul.
        
    # Simpler verification:
    # For each (i,j,k), T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return {"U": U_school, "V": V_school, "W": W_school}

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
