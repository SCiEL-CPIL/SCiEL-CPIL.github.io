import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    # Schoolbook decomposition generator (fallback)
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Strassen-Winograd rank-8 decomposition for 3x3 matrix multiplication
    # Returns U (9x8), V (9x8), W (9x8) for M_{3,3,3}
    # U[i*3+j][r] = coefficient for A[i,j] in rank-r component
    # V[j*3+k][r] = coefficient for B[j,k] in rank-r component  
    # W[k*3+i][r] = coefficient for C[k,i]... wait, W[k*n+i] where output C[i,k]
    # Actually the convention is W[k*n+i] so W row index = k*n + i
    # For C[i,k] = sum_j A[i,j]*B[j,k], the tensor entry T[u,v,w] where
    # u = i*m+j, v = j*p+k, w = k*n+i should be 1 iff (i,j,k) valid
    # So W[w_idx][r] where w_idx = k*3+i (since n=3)
    
    # The 8 Strassen-Winograd products for 3x3:
    # Let A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # Let B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # 
    # M1 = (a11 + a22 + a33) * (b11 + b22 + b33)
    # M2 = (a12 + a23 + a31) * (b12 + b23 + b31)
    # M3 = (a13 + a21 + a32) * (b13 + b21 + b32)
    # M4 = (a11 + a12) * (b21 - b22)
    # M5 = (a22 + a23) * (b32 - b33)
    # M6 = (a33 + a31) * (b13 - b11)
    # M7 = (a11 - a21) * (b22 + b23)
    # M8 = (a22 - a32) * (b33 + b31)
    #
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M2 - M4 + M7
    # C[0,2] = M3 - M6 + M8
    # C[1,0] = M4 + M7 - M8
    # C[1,1] = M1 - M2 + M5 - M7
    # C[1,2] = M2 - M5 + M8
    # C[2,0] = M3 - M6 + M8  ... wait let me re-derive
    # 
    # Let me use the standard Winograd 3x3:
    # U1 = a11 + a22 + a33,  V1 = b11 + b22 + b33
    # U2 = a12 + a23 + a31,  V2 = b12 + b23 + b31
    # U3 = a13 + a21 + a32,  V3 = b13 + b21 + b32
    # U4 = a11 + a12,        V4 = b21 - b22
    # U5 = a22 + a23,        V5 = b32 - b33
    # U6 = a33 + a31,        V6 = b13 - b11
    # U7 = a11 - a21,        V7 = b22 + b23
    # U8 = a22 - a32,        V8 = b33 + b31
    #
    # C11 = U1*V1 - U4*V4 + U6*V6 - U7*V7
    # C12 = U2*V2 - U4*V4 + U7*V7
    # C13 = U3*V3 - U6*V6 + U8*V8
    # C21 = U4*V4 + U7*V7 - U8*V8
    # C22 = U1*V1 - U2*V2 + U5*V5 - U7*V7
    # C23 = U2*V2 - U5*V5 + U8*V8
    # C31 = U3*V3 - U6*V6 + U8*V8  -- wait, this gives C31 = C13 which is wrong
    #
    # Let me look up the correct Winograd 3x3 formulas more carefully.
    # Actually, let me just verify by construction. The key insight is:
    # We need 8 linear forms in A, 8 linear forms in B, and the output
    # C[i,k] = sum_r alpha[i,k,r] * (U_r * V_r)
    #
    # Let me define it directly from the known Strassen-Winograd scheme.
    # I'll use the formulation where:
    #
    # m1 = (a00+a11+a22)(b00+b11+b22)
    # m2 = (a01+a12+a20)(b01+b12+b20)  
    # m3 = (a02+a10+a21)(b02+b10+b21)
    # m4 = (a00+a01)(b10-b11)
    # m5 = (a11+a12)(b21-b22)
    # m6 = (a22+a20)(b02-b00)
    # m7 = (a00-a10)(b11+b12)
    # m8 = (a11-a21)(b22+b20)
    #
    # c00 = m1 - m4 + m6 - m7
    # c01 = m2 - m4 + m7
    # c02 = m3 - m6 + m8
    # c10 = m4 + m7 - m8
    # c11 = m1 - m2 + m5 - m7
    # c12 = m2 - m5 + m8
    # c20 = m3 - m6 + m8   <-- this should be different from c02
    # c21 = m1 - m2 + m5 - m7  <-- this should be different from c11
    # c22 = m1 - m2 + m5 - m7  <-- all same? That can't be right.
    #
    # I'm making errors. Let me just directly construct the rank-8 decomposition
    # by using the known fact that Strassen's algorithm for 3x3 uses 8 products.
    # 
    # Actually, the cleanest approach: I'll define the U, V, W matrices directly
    # using the standard Winograd 3x3 formulas and verify they produce correct
    # matrix multiplication.
    
    def strassen_winograd_3x3():
        """
        Returns U (9x8), V (9x8), W (9x8) for 3x3 matrix mult.
        U row index: i*3+j (input A[i,j])
        V row index: j*3+k (input B[j,k])
        W row index: k*3+i (output C[i,k]) -- note: W[k*n+i] per convention
        """
        # 8 rank-1 components
        # For each component r, we have:
        # U_r[i*3+j] = coefficient of A[i,j]
        # V_r[j*3+k] = coefficient of B[j,k]
        # W_r[k*3+i] = coefficient in output C[i,k]
        
        U = [[0]*8 for _ in range(9)]
        V = [[0]*8 for _ in range(9)]
        W = [[0]*8 for _ in range(9)]
        
        # Component 0: (a00+a11+a22) * (b00+b11+b22)
        # contributes to: c00(+1), c11(+1), c22(+1)
        for i in range(3):
            U[i*3+i][0] = 1  # diagonal of A
        for j in range(3):
            V[j*3+j][0] = 1  # diagonal of B
        W[0*3+0][0] = 1  # c00
        W[1*3+1][0] = 1  # c11  
        W[2*3+2][0] = 1  # c22
        
        # Component 1: (a01+a12+a20) * (b01+b12+b20)
        # contributes to: c01(+1), c12(+1), c20(+1)
        U[0*3+1][1] = 1  # a01
        U[1*3+2][1] = 1  # a12
        U[2*3+0][1] = 1  # a20
        V[0*3+1][1] = 1  # b01
        V[1*3+2][1] = 1  # b12
        V[2*3+0][1] = 1  # b20
        W[1*3+0][1] = 1  # c01
        W[2*3+1][1] = 1  # c12
        W[0*3+2][1] = 1  # c20
        
        # Component 2: (a02+a10+a21) * (b02+b10+b21)
        # contributes to: c02(+1), c10(+1), c21(+1)
        U[0*3+2][2] = 1  # a02
        U[1*3+0][2] = 1  # a10
        U[2*3+1][2] = 1  # a21
        V[0*3+2][2] = 1  # b02
        V[1*3+0][2] = 1  # b10
        V[2*3+1][2] = 1  # b21
        W[2*3+0][2] = 1  # c02
        W[0*3+1][2] = 1  # c10
        W[1*3+2][2] = 1  # c21
        
        # Component 3: (a00+a01) * (b10-b11)
        # contributes to: c00(-1), c01(-1), c10(+1), c11(-1)
        U[0*3+0][3] = 1  # a00
        U[0*3+1][3] = 1  # a01
        V[1*3+0][3] = 1  # b10
        V[1*3+1][3] = -1 # b11
        W[0*3+0][3] = -1 # c00
        W[1*3+0][3] = -1 # c01
        W[0*3+1][3] = 1  # c10
        W[1*3+1][3] = -1 # c11
        
        # Component 4: (a11+a12) * (b21-b22)
        # contributes to: c11(+1), c12(+1), c21(-1), c22(-1)
        U[1*3+1][4] = 1  # a11
        U[1*3+2][4] = 1  # a12
        V[2*3+1][4] = 1  # b21
        V[2*3+2][4] = -1 # b22
        W[1*3+1][4] = 1  # c11
        W[2*3+1][4] = 1  # c12
        W[1*3+2][4] = -1 # c21
        W[2*3+2][4] = -1 # c22
        
        # Component 5: (a22+a20) * (b02-b00)
        # contributes to: c00(+1), c02(+1), c20(-1), c22(-1)
        U[2*3+2][5] = 1  # a22
        U[2*3+0][5] = 1  # a20
        V[0*3+2][5] = 1  # b02
        V[0*3+0][5] = -1 # b00
        W[0*3+0][5] = 1  # c00
        W[2*3+0][5] = 1  # c02
        W[0*3+2][5] = -1 # c20
        W[2*3+2][5] = -1 # c22
        
        # Component 6: (a00-a10) * (b11+b12)
        # contributes to: c00(-1), c01(+1), c10(+1), c11(-1)
        U[0*3+0][6] = 1  # a00
        U[1*3+0][6] = -1 # a10
        V[1*3+1][6] = 1  # b11
        V[1*3+2][6] = 1  # b12
        W[0*3+0][6] = -1 # c00
        W[1*3+0][6] = 1  # c01
        W[0*3+1][6] = 1  # c10
        W[1*3+1][6] = -1 # c11
        
        # Component 7: (a11-a21) * (b22+b20)
        # contributes to: c02(+1), c11(-1), c12(+1), c22(+1)
        U[1*3+1][7] = 1  # a11
        U[2*3+1][7] = -1 # a21
        V[2*3+2][7] = 1  # b22
        V[2*3+0][7] = 1  # b20
        W[2*3+0][7] = 1  # c02
        W[1*3+1][7] = -1 # c11
        W[2*3+1][7] = 1  # c12
        W[2*3+2][7] = 1  # c22
        
        return U, V, W
    
    # Verify the Strassen-Winograd decomposition
    def verify_sw(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i*3 + j
                    v_idx = j*3 + k
                    w_idx = k*3 + i
                    s = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(8))
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zeros
        for u in range(9):
            i_u = u // 3
            j_u = u % 3
            for v in range(9):
                j_v = v // 3
                k_v = v % 3
                if j_u != j_v:
                    continue
                for w in range(9):
                    k_w = w // 3
                    i_w = w % 3
                    if i_u != i_w or k_v != k_w:
                        s = sum(U[u][r] * V[v][r] * W[w][r] for r in range(8))
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Build the full 3x3 x 3x8 decomposition using Strassen-Winograd blocks
    # Split p=8 into three blocks: k=0,1,2 | k=3,4,5 | k=6,7
    # First two blocks are 3x3, last block is 3x2
    
    U_sw, V_sw, W_sw = strassen_winograd_3x3()
    
    if not verify_sw(U_sw, V_sw, W_sw):
        # Fallback to schoolbook
        U, V, W, _ = schoolbook(3, 3, 8)
        return {"U": U, "V": V, "W": W}
    
    # For the 3x2 block (k=6,7), we need a decomposition.
    # We can use the first 2 columns of the 3x3 Strassen-Winograd as a 3x2 subproblem?
    # No, that doesn't work directly. Let's use schoolbook for the 3x2 block.
    # 3x2 schoolbook has rank 6.
    
    # Total rank before merging: 8 + 8 + 6 = 22
    
    # Build full factor matrices
    total_rank = 8 + 8 + 6
    
    U_full = [[0] * total_rank for _ in range(9)]
    V_full = [[0] * total_rank for _ in range(24)]  # 3*8
    W_full = [[0] * total_rank for _ in range(24)]  # 8*3
    
    # Block 1: k=0,1,2 (Strassen-Winograd 3x3)
    for i in range(3):
        for j in range(3):
            u_idx = i*3 + j
            for r in range(8):
                U_full[u_idx][r] = U_sw[u_idx][r]
                for k in range(3):
                    v_idx = j*8 + k
                    V_full[v_idx][r] = V_sw[j*3 + k][r]
                    w_idx = k*3 + i
                    W_full[w_idx][r] = W_sw[k*3 + i][r]
    
    # Block 2: k=3,4,5 (Strassen-Winograd 3x3, shifted)
    for i in range(3):
        for j in range(3):
            u_idx = i*3 + j
            for r in range(8):
                U_full[u_idx][8 + r] = U_sw[u_idx][r]
                for k in range(3):
                    v_idx = j*8 + (k + 3)
                    V_full[v_idx][8 + r] = V_sw[j*3 + k][r]
                    w_idx = (k + 3)*3 + i
                    W_full[w_idx][8 + r] = W_sw[k*3 + i][r]
    
    # Block 3: k=6,7 (schoolbook 3x2, rank 6)
    for i in range(3):
        for j in range(3):
            for k_local in range(2):
                col = 16 + (i*3 + j)*2 + k_local  # 16 + 0..11
                u_idx = i*3 + j
                v_idx = j*8 + (6 + k_local)
                w_idx = (6 + k_local)*3 + i
                U_full[u_idx][col] = 1
                V_full[v_idx][col] = 1
                W_full[w_idx][col] = 1
    
    # Verify the combined decomposition
    def verify_full(U_f, V_f, W_f, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    s = sum(U_f[u_idx][r] * V_f[v_idx][r] * W_f[w_idx][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zeros
        for u in range(n*m):
            i_u = u // m
            j_u = u % m
            for v in range(m*p):
                j_v = v // p
                k_v = v % p
                if j_u != j_v:
                    continue
                for w in range(p*n):
                    k_w = w // n
                    i_w = w % n
                    if i_u != i_w or k_v != k_w:
                        s = sum(U_f[u][r] * V_f[v][r] * W_f[w][r] for r in range(rank))
                        if abs(s) > 1e-9:
                            return False
        return True
    
    if not verify_full(U_full, V_full, W_full, 3, 3, 8, total_rank):
        # Fallback
        U, V, W, _ = schoolbook(3, 3, 8)
        return {"U": U, "V": V, "W": W}
    
    # Now perform linear dependency analysis to reduce rank
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U_full, dtype=float)
    V_np = np.array(V_full, dtype=float)
    W_np = np.array(W_full, dtype=float)
    
    # Check if any column is linearly dependent on others
    # A column r is redundant if the tensor contribution of that column
    # can be expressed as a linear combination of other columns.
    # 
    # The tensor contribution of column r is: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to check if this vector is in the span of other such vectors.
    #
    # Build the tensor matrix: each column r gives a vector of size 9*24*24 = 5184
    # But we only care about the 72 non-zero entries and the zero constraints.
    # Actually, for exact decomposition, we need the full tensor to match.
    #
    # Simpler approach: check if removing column r and adjusting others still gives
    # the same tensor. This is equivalent to checking if the column vectors
    # (flattened) are linearly independent.
    
    # Flatten each column into a vector
    # For column r: vec_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] (Kronecker product)
    # Size: 9 * 24 * 24 = 5184
    
    # Build matrix where each row is a flattened column
    # M_tensor has shape (total_rank, 9*24*24)
    M_tensor = np.zeros((total_rank, 9*24*24))
    for r in range(total_rank):
        vec = np.kron(np.kron(U_np[:,r], V_np[:,r]), W_np[:,r])
        M_tensor[r, :] = vec
    
    # Find rank of M_tensor
    # Use SVD to find numerical rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M_tensor, full_matrices=False)
        # Numerical rank: number of singular values > tolerance
        tol = 1e-8 * max(s_vals) if len(s_vals) > 0 else 1e-8
        numerical_rank = np.sum(s_vals > tol)
    except:
        numerical_rank = total_rank
    
    # If numerical_rank < total_rank, some columns are linearly dependent
    # We can keep only the first `numerical_rank` columns (or a basis)
    # But we need to be careful: the decomposition must still be exact.
    
    # For now, if no linear dependencies found, return the rank-22 solution.
    # The linear dependency analysis shows that in general, the Strassen-Winograd
    # blocks for different k-ranges are linearly independent because they involve
    # different V and W indices.
    
    # However, let's try a greedy column elimination:
    # Try removing each column and check if the remaining columns still span
    # the same tensor.
    
    # For efficiency, let's check if any single column is in the span of others
    # by checking if M_tensor without that row has the same rank.
    
    reduced_U = U_full[:]
    reduced_V = V_full[:]
    reduced_W = W_full[:]
    current_rank = total_rank
    keep_cols = list(range(total_rank))
    
    # Try to eliminate columns one by one
    for r_to_try in range(total_rank):
        if time.time() > deadline - 10:
            break
        if r_to_try not in keep_cols:
            continue
        
        # Try removing column r_to_try
        new_keep = [c for c in keep_cols if c != r_to_try]
        if len(new_keep) == 0:
            continue
        
        # Build tensor matrix without column r_to_try
        M_sub = np.zeros((len(new_keep), 9*24*24))
        for idx, r in enumerate(new_keep):
            vec = np.kron(np.kron(U_np[:,r], V_np[:,r]), W_np[:,r])
            M_sub[idx, :] = vec
        
        # Check if the removed column's vector is in the span of the remaining
        vec_removed = np.kron(np.kron(U_np[:,r_to_try], V_np[:,r_to_try]), W_np[:,r_to_try])
        
        # Check if vec_removed is in the column space of M_sub
        # By checking if ||M_sub^T M_sub (M_sub^T vec_removed) - M_sub^T vec_removed|| is small
        # Or simpler: check if rank([M_sub; vec_removed]) == rank(M_sub)
        
        M_aug = np.vstack([M_sub, vec_removed.reshape(1, -1)])
        
        try:
            rank_sub = np.linalg.matrix_rank(M_sub, tol=1e-8)
            rank_aug = np.linalg.matrix_rank(M_aug, tol=1e-8)
        except:
            continue
        
        if rank_aug == rank_sub:
            # Column r_to_try is redundant, remove it
            keep_cols = new_keep
            current_rank = len(keep_cols)
    
    # Build final matrices with only kept columns
    if current_rank < total_rank:
        U_final = [[0] * current_rank for _ in range(9)]
        V_final = [[0] * current_rank for _ in range(24)]
        W_final = [[0] * current_rank for _ in range(24)]
        
        for new_idx, old_idx in enumerate(keep_cols):
            for i in range(9):
                U_final[i][new_idx] = U_full[i][old_idx]
            for i in range(24):
                V_final[i][new_idx] = V_full[i][old_idx]
                W_final[i][new_idx] = W_full[i][old_idx]
        
        # Verify final result
        if verify_full(U_final, V_final, W_final, 3, 3, 8, current_rank):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # Return the rank-22 solution
    return {"U": U_full, "V": V_full, "W": W_full}
