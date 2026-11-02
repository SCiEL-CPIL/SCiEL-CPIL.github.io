import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150.0

    n, m, p = 3, 4, 4
    # Shapes:
    # U: (n*m, rank) -> (12, rank)
    # V: (m*p, rank) -> (16, rank)
    # W: (p*n, rank) -> (12, rank)
    # Trace convention: W[k*n+i, r] corresponds to C[i, k]

    def verify(U, V, W):
        # U: list of lists or numpy array
        # Convert to numpy for speed if needed, but lists are fine for small sizes
        num_rows_u = len(U)
        num_rows_v = len(V)
        num_rows_w = len(W)
        rank = len(U[0])
        
        for a in range(num_rows_u):
            for b in range(num_rows_v):
                for c in range(num_rows_w):
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    
                    # Check if (a,b,c) is a valid multiplication index
                    # a = i*m + j
                    # b = j*p + k
                    # c = k*n + i
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        if val != 1:
                            return False
                    else:
                        if val != 0:
                            return False
        return True

    # Rank 44 Construction
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        # a_r, a_c: start row, start col of A block (2x2)
        # b_r, b_c: start row, start col of B block (2x2)
        # c_r, c_c: start row, start col of C block (2x2)
        
        # Local indices mapping:
        # A: (r, c) -> a_r+m, a_c+m
        # B: (r, c) -> b_r*m, b_c+m
        # C: (r, c) -> c_r*m, c_c+m
        
        # Update 1: M1
        U[(a_r)*m + a_c][start_col + 0] += 1
        U[(a_r+1)*m + (a_c+1)][start_col + 0] += 1
        V[(b_r)*p + b_c][start_col + 0] += 1
        V[(b_r+1)*p + (b_c+1)][start_col + 0] += 1
        W[(c_c)*n + c_r][start_col + 0] += 1 # C_00
        W[(c_c+1)*n + (c_r+1)][start_col + 0] += 1 # C_11
        
        # Update 2: M2
        U[(a_r+1)*m + a_c][start_col + 1] += 1 # A_10
        U[(a_r+1)*m + (a_c+1)][start_col + 1] += 1 # A_11
        V[(b_r)*p + b_c][start_col + 1] += 1 # B_00
        W[(c_c)*n + (c_r+1)][start_col + 1] += 1 # C_10
        W[(c_c+1)*n + (c_r+1)][start_col + 1] -= 1 # C_11
        
        # Update 3: M3
        U[(a_r)*m + a_c][start_col + 2] += 1 # A_00
        V[(b_r+1)*p + b_c][start_col + 2] += 1 # B_10
        V[(b_r+1)*p + (b_c+1)][start_col + 2] += 1 # B_11
        W[(c_c+1)*n + c_r][start_col + 2] += 1 # C_01
        W[(c_c+1)*n + (c_r+1)][start_col + 2] += 1 # C_11
        
        # Update 4: M4
        U[(a_r+1)*m + (a_c+1)][start_col + 3] += 1 # A_11
        V[(b_r)*p + b_c][start_col + 3] -= 1 # B_00
        V[(b_r)*p + (b_c+1)][start_col + 3] += 1 # B_01
        W[(c_c)*n + c_r][start_col + 3] += 1 # C_00
        W[(c_c)*n + (c_r+1)][start_col + 3] += 1 # C_10
        
        # Update 5: M5
        U[(a_r)*m + a_c][start_col + 4] += 1 # A_00
        U[(a_r)*m + (a_c+1)][start_col + 4] += 1 # A_01
        V[(b_r+1)*p + (b_c+1)][start_col + 4] += 1 # B_11
        W[(c_c)*n + c_r][start_col + 4] -= 1 # C_00
        W[(c_c+1)*n + c_r][start_col + 4] += 1 # C_01
        
        # Update 6: M6
        U[(a_r+1)*m + a_c][start_col + 5] += 1 # A_10
        U[(a_r)*m + a_c][start_col + 5] -= 1 # A_00
        V[(b_r)*p + b_c][start_col + 5] += 1 # B_00
        V[(b_r)*p + (b_c+1)][start_col + 5] += 1 # B_01
        W[(c_c+1)*n + (c_r+1)][start_col + 5] += 1 # C_11
        
        # Update 7: M7
        U[(a_r)*m + (a_c+1)][start_col + 6] += 1 # A_01
        U[(a_r+1)*m + (a_c+1)][start_col + 6] -= 1 # A_11
        V[(b_r+1)*p + b_c][start_col + 6] += 1 # B_10
        V[(b_r+1)*p + (b_c+1)][start_col + 6] += 1 # B_11
        W[(c_c)*n + c_r][start_col + 6] += 1 # C_00
        
        return start_col + 7

    def add_trivial_block(a_r, b_c_start, start_col):
        # Computes C[a_r, b_c_start:b_c_start+2] = A[a_r, :] * B[:, b_c_start:b_c_start+2]
        # A is 1x4, B is 4x2
        # 8 multiplications
        c = start_col
        for j in range(4):
            # Col 0 of the 2-col block
            U[a_r*m + j][c] += 1
            V[j*p + b_c_start][c] += 1
            W[b_c_start*n + a_r][c] += 1
            c += 1
            # Col 1 of the 2-col block
            U[a_r*m + j][c] += 1
            V[j*p + b_c_start + 1][c] += 1
            W[(b_c_start + 1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (Cols 0,1 of C)
    # C_0:2, 0:2
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col) # A_0:2,0:2 * B_0:2,0:2
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col) # A_0:2,2:4 * B_2:4,0:2
    # C_2, 0:2
    col = add_trivial_block(2, 0, col)

    # Right Side (Cols 2,3 of C)
    # C_0:2, 2:4
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col) # A_0:2,0:2 * B_0:2,2:4
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col) # A_0:2,2:4 * B_2:4,2:4
    # C_2, 2:4
    col = add_trivial_block(2, 2, col)

    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to Schoolbook (Rank 48)
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m + j][r] = 1
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        return {"U": U, "V": V, "W": W}
