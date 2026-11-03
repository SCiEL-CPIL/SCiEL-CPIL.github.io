import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Split middle dim m=4 into two 2s: m1=2, m2=2
    # Split right dim p=6 into three 2s: p1=2, p2=2, p3=2
    # This gives 6 blocks of 3x2x2
    
    # For each 3x2x2 block, use:
    #   - Strassen 2x2x2 for the first 2 rows of A (rank 7)
    #   - 2 schoolbook terms for the 3rd row of A (rank 2)
    # Total per block: rank 9
    # Total: 6 * 9 = 54
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 54
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 54
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 54
    
    # Strassen 2x2x2 decomposition (rank 7)
    # U2[i,c], V2[j,c], W2[k,c] for i=(a,b), j=(b,d), k=(d,a) in 2x2
    # Standard Strassen:
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B10-B11)
    # P4: A11(B01-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # For 2x2x2: U rows indexed by (a,b) -> row a*2+b
    # V rows indexed by (b,d) -> row b*2+d
    # W rows indexed by (d,a) -> row d*2+a
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # row 0: (a=0,b=0)
        [0, 0, 0, 0, 1, 0, 1],   # row 1: (a=0,b=1)
        [0, 1, 0, 0, 0, 1, 0],   # row 2: (a=1,b=0)
        [1, 1, 0, 1, 0, 0, -1],  # row 3: (a=1,b=1)
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # row 0: (b=0,d=0)
        [0, 0, 0, 1, 0, 1, 0],   # row 1: (b=0,d=1)
        [0, 0, 1, 0, 0, 0, 1],   # row 2: (b=1,d=0)
        [1, 0, -1, 0, 1, 0, 1],  # row 3: (b=1,d=1)
    ]
    strassen_W = [
        [1, 0, 1, 0, 1, -1, -1],  # row 0: (d=0,a=0)
        [1, 0, -1, 0, 1, -1, 1],  # row 1: (d=0,a=1)
        [0, 1, 0, -1, 0, 1, 1],   # row 2: (d=1,a=0)
        [1, 1, 0, 1, 0, 1, 1],    # row 3: (d=1,a=1)
    ]
    
    # Verify Strassen 2x2x2:
    # C[0,0] = sum_c U[0,c]*V[?,c]*W[0,c]... 
    # Actually let me just trust the standard formula and verify at the end.
    
    # For each block (i_block, j_block) where i_block in {0,1} (middle split), j_block in {0,1,2} (right split)
    # Block: A[:, i_block*2:(i_block+1)*2] * B[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2]
    # This is 3x2 times 2x2, giving 3x2 output
    
    col_offset = 0
    for i_block in range(2):  # split of middle dimension
        for j_block in range(3):  # split of right dimension
            # This block contributes to C[:, j_block*2:(j_block+1)*2]
            # A rows: all 3 rows, cols i_block*2, i_block*2+1
            # B rows: i_block*2, i_block*2+1, cols j_block*2, j_block*2+1
            
            # Strassen part for rows 0,1 of A (2x2x2 subproblem)
            for c in range(7):
                # U: rows (a,b) where a in {0,1}, b in {0,1}
                # A row a, col i_block*2 + b
                for a in range(2):
                    for b in range(2):
                        u_val = strassen_U[a*2+b][c]
                        if u_val != 0:
                            U[a * m + (i_block*2 + b)][col_offset] += u_val
                
                # V: rows (b,d) where b in {0,1}, d in {0,1}
                # B row i_block*2 + b, col j_block*2 + d
                for b in range(2):
                    for d in range(2):
                        v_val = strassen_V[b*2+d][c]
                        if v_val != 0:
                            V[(i_block*2 + b) * p + (j_block*2 + d)][col_offset] += v_val
                
                # W: rows (d,a) where d in {0,1}, a in {0,1}
                # C row a, col j_block*2 + d
                for d in range(2):
                    for a in range(2):
                        w_val = strassen_W[d*2+a][c]
                        if w_val != 0:
                            W[(j_block*2 + d) * n + a][col_offset] += w_val
            
            col_offset += 7
            
            # Schoolbook part for row 2 of A (1x2x2)
            # C[2, j_block*2+0] = A[2, i_block*2+0]*B[i_block*2+0, j_block*2+0] + A[2, i_block*2+1]*B[i_block*2+1, j_block*2+0]
            # C[2, j_block*2+1] = A[2, i_block*2+0]*B[i_block*2+0, j_block*2+1] + A[2, i_block*2+1]*B[i_block*2+1, j_block*2+1]
            
            # Term 1: A[2, i_block*2+0] * B[i_block*2+0, :]
            # This contributes to C[2, j_block*2+0] and C[2, j_block*2+1]
            U[2 * m + (i_block*2 + 0)][col_offset] = 1
            V[(i_block*2 + 0) * p + (j_block*2 + 0)][col_offset] = 1
            W[(j_block*2 + 0) * n + 2][col_offset] = 1
            V[(i_block*2 + 0) * p + (j_block*2 + 1)][col_offset] = 1
            W[(j_block*2 + 1) * n + 2][col_offset] = 1
            
            col_offset += 1
            
            # Term 2: A[2, i_block*2+1] * B[i_block*2+1, :]
            U[2 * m + (i_block*2 + 1)][col_offset] = 1
            V[(i_block*2 + 1) * p + (j_block*2 + 0)][col_offset] = 1
            W[(j_block*2 + 0) * n + 2][col_offset] = 1
            V[(i_block*2 + 1) * p + (j_block*2 + 1)][col_offset] = 1
            W[(j_block*2 + 1) * n + 2][col_offset] = 1
            
            col_offset += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For random test matrices
    for trial in range(5):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute using decomposition
        C_decomposed = np.zeros((n, p))
        for c in range(total_rank):
            u_vec = U_arr[:, c]  # (n*m,)
            v_vec = V_arr[:, c]  # (m*p,)
            w_vec = W_arr[:, c]  # (p*n,)
            
            # u_vec[i*m+j] = U[i,j]
            # v_vec[j*p+k] = V[j,k]
            # w_vec[k*n+i] = W[k,i]
            
            # C[i,k] += sum_j U[i,j] * V[j,k] * W[k,i]
            U_mat = u_vec.reshape(n, m)
            V_mat = v_vec.reshape(m, p)
            W_mat = w_vec.reshape(p, n)
            
            # C[i,k] += sum_j U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            # = (U_mat @ V_mat) element-wise times... no
            # C[i,k] += sum_j U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            # This is not a standard matrix product.
            # C[i,k] = sum_j U[i,j]*V[j,k]*W[k,i]
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
                    C_decomposed[i, k] += s
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            raise ValueError(f"Decomposition failed for trial {trial}")
    
    return {"U": U, "V": V, "W": W}

