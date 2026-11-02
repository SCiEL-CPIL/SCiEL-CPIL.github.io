import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_idx, b_idx, c_idx, start_col):
        a, b, c, d = a_idx
        e, f, g, h = b_idx
        c11, c12, c21, c22 = c_idx
        
        # M1 = (a+d)(e+h)
        U[a][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1; W[c22][start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1; V[h][start_col] -= 1
        W[c12][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1; V[e][start_col] -= 1
        W[c11][start_col] += 1; W[c21][start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a][start_col] += 1; U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1; W[c12][start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c][start_col] += 1; U[a][start_col] -= 1
        V[e][start_col] += 1; V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b][start_col] += 1; U[d][start_col] -= 1
        V[g][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col

    # Iterate over the 3 blocks of width 2 in the p dimension
    for k in range(3):
        # A_L: cols 0,1 of A (indices 0,1,4,5)
        # A_R: cols 2,3 of A (indices 2,3,6,7)
        
        # B_Tk: rows 0,1 of B, cols 2k, 2k+1
        # B_Bk: rows 2,3 of B, cols 2k, 2k+1
        
        # C indices for cols 2k, 2k+1
        # C is p x n = 6 x 2. Index = row * n + col.
        # Row 0, Col 2k -> 0*2 + 2k = 2k
        # Row 0, Col 2k+1 -> 0*2 + 2k+1 = 2k+1
        # Row 1, Col 2k -> 1*2 + 2k = 2k+2
        # Row 1, Col 2k+1 -> 1*2 + 2k+1 = 2k+3
        
        # Wait, the task says W[k*n+i, r].
        # Usually C is stored row-major.
        # Let's check the previous program's C_idx.
        # Previous: c11 = 2k*2 + 0. This is (2k)*2.
        # This implies C index = row * n + col?
        # If C is 2x6 (n x p), index = row * p + col.
        # But W shape is (p*n, rank).
        # So W index corresponds to C flattened in column-major? Or row-major?
        # "W[k*n+i, r]". k is likely the row of C (0..p-1), i is col of C (0..n-1).
        # So W index = k*n + i.
        # C[k, i] = sum_j A[i, j] B[j, k].
        # Here A is 2x4, B is 4x6, C is 2x6.
        # So C[i, k] where i in 0..1, k in 0..5.
        # W index = k*n + i = k*2 + i.
        
        # Let's re-verify the indices for C.
        # For block k (cols 2k, 2k+1 of B, and thus cols 2k, 2k+1 of C? No.)
        # C[i, col] depends on A[i, j] and B[j, col].
        # We split B into 3 blocks of 2 cols.
        # Block 0: B cols 0,1. Contributes to C cols 0,1.
        # Block 1: B cols 2,3. Contributes to C cols 2,3.
        # Block 2: B cols 4,5. Contributes to C cols 4,5.
        
        # For block k, C cols are 2k, 2k+1.
        # C[0, 2k] index: row=0, col=2k. W index = (2k)*2 + 0 = 4k.
        # C[0, 2k+1] index: row=0, col=2k+1. W index = (2k+1)*2 + 0 = 4k+2.
        # C[1, 2k] index: row=1, col=2k. W index = (2k)*2 + 1 = 4k+1.
        # C[1, 2k+1] index: row=1, col=2k+1. W index = (2k+1)*2 + 1 = 4k+3.
        
        # Previous program used:
        # c11 = 2k*2 + 0 = 4k. (C[0, 2k])
        # c12 = (2k+1)*2 + 0 = 4k+2. (C[0, 2k+1])
        # c21 = 2k*2 + 1 = 4k+1. (C[1, 2k])
        # c22 = (2k+1)*2 + 1 = 4k+3. (C[1, 2k+1])
        
        # This matches.
        
        A_L_idx = [0, 1, 4, 5]
        A_R_idx = [2, 3, 6, 7]
        
        # B indices: row * p + col.
        # B_Tk: rows 0,1. Cols 2k, 2k+1.
        # e=B[0,2k], f=B[0,2k+1], g=B[1,2k], h=B[1,2k+1]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        
        # B_Bk: rows 2,3. Cols 2k, 2k+1.
        # e=B[2,2k], f=B[2,2k+1], g=B[3,2k], h=B[3,2k+1]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        C_idx = [4*k, 4*k+2, 4*k+1, 4*k+3]
        
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
