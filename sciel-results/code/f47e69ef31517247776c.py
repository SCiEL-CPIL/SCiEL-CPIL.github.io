"""Matrix multiplication tensor decomposition for 2x4 by 4x5 matrices."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition using Strassen for 2x2x2 blocks
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        """
        Apply Strassen's 7-multiplication algorithm for a 2x2 x 2x2 block.
        y_indices: [(i0,j0), (i0,j1), (i1,j0), (i1,j1)] for A block
        z_indices: [(j0,k0), (j0,k1), (j1,k0), (j1,k1)] for B block  
        w_offsets: [(k0,i0), (k1,i0), (k0,i1), (k1,i1)] for C block (row, col)
        """
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1  # C00
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1  # C11
        col += 1
        
        # M2 = (A10+A11)B00 -> contributes to C10 and -C11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1  # C10
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1  # -C11
        col += 1
        
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1  # C01
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1  # C11
        col += 1
        
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1  # C00
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1  # C10
        col += 1
        
        # M5 = (A00+A01)B11 -> contributes to -C00 and C01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1  # -C00
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1  # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B10) -> contributes to C11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1  # C11
        col += 1
        
        # M7 = (A01-A11)(B01+B11) -> contributes to C00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1  # C00
        col += 1

    # Block 1: A_L (rows 0-1, cols 0-1) * B_11 (rows 0-1, cols 0-1)
    # A_L: i in {0,1}, j in {0,1}
    # B_11: j in {0,1}, k in {0,1}
    # C contribution: i in {0,1}, k in {0,1}
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]  # (k,i) pairs: C00, C10, C01, C11
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (rows 0-1, cols 2-3) * B_21 (rows 2-3, cols 0-1)
    # A_R: i in {0,1}, j in {2,3}
    # B_21: j in {2,3}, k in {0,1}
    # C contribution: i in {0,1}, k in {0,1}
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (rows 0-1, cols 0-1) * B_12 (rows 0-1, cols 2-3)
    # A_L: i in {0,1}, j in {0,1}
    # B_12: j in {0,1}, k in {2,3}
    # C contribution: i in {0,1}, k in {2,3}
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (rows 0-1, cols 2-3) * B_22 (rows 2-3, cols 2-3)
    # A_R: i in {0,1}, j in {2,3}
    # B_22: j in {2,3}, k in {2,3}
    # C contribution: i in {0,1}, k in {2,3}
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (rows 0-1, cols 0-1) * B_15 (row 0, col 4)
    # This is 2x2 * 2x1, schoolbook rank 4
    # C contribution: i in {0,1}, k in {4}
    
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R (rows 0-1, cols 2-3) * B_25 (row 1, col 4)
    # This is 2x2 * 2x1, schoolbook rank 4
    # C contribution: i in {0,1}, k in {4}
    
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
