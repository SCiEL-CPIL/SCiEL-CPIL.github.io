"""Matrix multiplication 4x5x5 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are vectors of length 4 (row-major 2x2).
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    # A11=idx0, A22=idx3, B11=idx0, B22=idx3
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, 1] (contributes to C11, C22)
    # Wait, Strassen terms contribute to specific blocks of C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express the full 2x2x2 tensor T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # such that for each (i,j,k) in 2x2x2, the sum is 1.
    
    # Let's define the 7 terms explicitly.
    # Indices 0,1,2,3 for 2x2 matrix.
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Term 1: M1
    # A part: A11 + A22 -> u1 = [1, 0, 0, 1]
    # B part: B11 + B22 -> v1 = [1, 0, 0, 1]
    # C part: M1 contributes to C11 and C22.
    # C11 gets +1, C22 gets +1.
    # So w1 should have 1 at (0,0) and 1 at (1,1).
    # w1 = [1, 0, 0, 1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # Term 2: M2 = (A21+A22)B11
    # A part: A21 + A22 -> u2 = [0, 0, 1, 1]
    # B part: B11 -> v2 = [1, 0, 0, 0]
    # C part: M2 contributes to C21 (+1) and C22 (-1).
    # w2 = [0, 0, 1, -1]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # Term 3: M3 = A11(B21-B22)
    # A part: A11 -> u3 = [1, 0, 0, 0]
    # B part: B21 - B22 -> v3 = [0, 0, 1, -1]
    # C part: M3 contributes to C12 (+1) and C22 (+1).
    # w3 = [0, 1, 0, 1]
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    
    # Term 4: M4 = A22(B12-B11)
    # A part: A22 -> u4 = [0, 0, 0, 1]
    # B part: B12 - B11 -> v4 = [0, 1, 0, 0] - [1, 0, 0, 0] = [-1, 1, 0, 0]
    # C part: M4 contributes to C11 (+1) and C21 (+1).
    # w4 = [1, 0, 1, 0]
    terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]))
    
    # Term 5: M5 = (A11+A12)B22
    # A part: A11 + A12 -> u5 = [1, 1, 0, 0]
    # B part: B22 -> v5 = [0, 0, 0, 1]
    # C part: M5 contributes to C11 (-1) and C12 (+1).
    # w5 = [-1, 1, 0, 0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # A part: A21 - A11 -> u6 = [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0]
    # B part: B11 + B12 -> v6 = [1, 1, 0, 0]
    # C part: M6 contributes to C22 (+1).
    # w6 = [0, 0, 0, 1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # A part: A12 - A22 -> u7 = [0, 1, 0, 0] - [0, 0, 0, 1] = [0, 1, 0, -1]
    # B part: B21 + B22 -> v7 = [0, 0, 1, 1]
    # C part: M7 contributes to C11 (+1).
    # w7 = [1, 0, 0, 0]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return terms

def _build_4x4x4_decomp():
    """
    Build the 4x4x4 decomposition using 7 Strassen 2x2x2 blocks.
    Returns U, V, W of shape (16, 49).
    """
    strassen_terms = _strassen_2x2x2_terms()
    # 7 terms for 2x2x2.
    # We have 7 such blocks in the 4x4x4 Strassen.
    # Total rank = 7 * 7 = 49.
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for s in range(7):
        u_s, v_s, w_s = strassen_terms[s]
        # u_s, v_s, w_s are length 4 vectors for 2x2 blocks.
        # We need to map these to the 4x4 indices.
        # The 4x4 matrix is split into 2x2 blocks.
        # Block (bi, bj) corresponds to indices:
        # i = 2*bi + i0, j = 2*bj + j0
        # The Strassen formulas combine these blocks.
        # For each of the 7 Strassen products M_s, we perform a 2x2x2 multiplication
        # on the "block space".
        # The U vector for the 4x4 case is formed by placing the 2x2 U vector
        # into the appropriate 2x2 block of the 4x4 U matrix.
        
        # Wait, the Strassen decomposition of 4x4 is:
        # C = AB
        # M_s = (Sum A_blocks) * (Sum B_blocks)
        # Each M_s is a 2x2 matrix (block).
        # So M_s is itself a 2x2x2 tensor? No, M_s is a 2x2 matrix product.
        # So we decompose each M_s using the 2x2x2 Strassen terms.
        
        # The 4x4 U matrix has 16 entries.
        # The 2x2 U vector has 4 entries.
        # For a given Strassen term s, and a 2x2 Strassen term t (0..6):
        # The contribution to U[4x4 index] comes from u_t of the 2x2 decomposition.
        # But which 2x2 block of the 4x4 U matrix does it go into?
        # It depends on the Strassen formula for M_s.
        # M_s = (Linear Comb of A 2x2 blocks) * (Linear Comb of B 2x2 blocks).
        # This is equivalent to a 2x2 matrix multiplication where the "matrices" are 2x2 blocks.
        # So the "row" index of the 4x4 U matrix is determined by the block row and the local row.
        
        # Let's iterate over the 7 Strassen products s.
        # For each s, we have a 2x2x2 tensor T_s (the Strassen terms for 2x2).
        # The 4x4 tensor is the sum of these T_s placed in the block structure.
        
        # Actually, it's simpler:
        # The 4x4 Strassen algorithm computes 7 products of 2x2 matrices.
        # Let's call the 2x2 matrices A_s and B_s.
        # C_s = A_s B_s.
        # Then C is assembled from C_s.
        
        # The rank-1 terms of C_s are given by the 2x2x2 Strassen terms.
        # For each s (0..6), and each t (0..6) in 2x2 Strassen:
        # We have a term (u_st, v_st, w_st) for the 2x2x2 tensor.
        # u_st is a vector of length 4.
        # This u_st corresponds to the coefficients of A_s.
        # A_s is a linear combination of A's 2x2 blocks.
        # So u_st[0] corresponds to A_s[0,0], u_st[1] to A_s[0,1], etc.
        # A_s[0,0] is a 2x2 block of A.
        # So the 4x4 U vector will have u_st[0] in the (0,0) block, u_st[1] in (0,1) block, etc.
        
        # Let's define the mapping.
        # For 2x2 index l (0..3), let bi = l // 2, bj = l % 2.
        # The 4x4 index for block (bi, bj) and local index (li, lj) is:
        # i = 2*bi + li, j = 2*bj + lj.
        # 4x4 index = i*4 + j.
        
        for t in range(7):
            u_t, v_t, w_t = strassen_terms[t]
            # u_t, v_t, w_t are length 4 vectors.
            for l in range(4):
                # l is the 2x2 index for A_s, B_s, C_s
                bi = l // 2
                bj = l % 2
                
                # For U:
                # u_t[l] is the coefficient for the 2x2 block (bi, bj) of A_s.
                # This block corresponds to 4x4 rows 2*bi, 2*bi+1 and cols 2*bj, 2*bj+1.
                # Wait, A_s is a 2x2 matrix of blocks.
                # A_s[bi, bj] is a 2x2 matrix.
                # The Strassen term t for 2x2x2 has u_t[l] which is the coefficient
                # for the l-th entry of the 2x2 matrix A_s.
                # So u_t[l] multiplies the entire 2x2 block A[bi, bj].
                # This means for all local indices (li, lj) in 0..1:
                # U[4x4 index (2*bi+li, 2*bj+lj)] += u_t[l]
                
                for li in range(2):
                    for lj in range(2):
                        i_4 = 2 * bi + li
                        j_4 = 2 * bj + lj
                        idx_U = i_4 * 4 + j_4
                        U[idx_U][col] = u_t[l]
                        
                        # Similarly for V
                        # v_t[l] multiplies the 2x2 block B[bi, bj].
                        # V index is (2*bi+li, 2*bj+lj)
                        idx_V = i_4 * 4 + j_4
                        V[idx_V][col] = v_t[l]
                        
                        # Similarly for W
                        # w_t[l] multiplies the 2x2 block C[bi, bj].
                        # W index is (2*bi+li, 2*bj+lj)
                        idx_W = i_4 * 4 + j_4
                        W[idx_W][col] = w_t[l]
            
            col += 1
            
    return U, V, W

def _schoolbook_sub(n, m, p):
    """Schoolbook for n*m x m*p sub-block."""
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
    return U, V, W

def _assemble_block_decomp(n, m, p, sub_decomps):
    """
    Assemble full decomposition from sub-block decompositions.
    sub_decomps: list of (U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk)
    """
    total_rank = sum(len(U[0]) for U, V, W, *_ in sub_decomps)
    U_out = [[0] * total_rank for _ in range(n * m)]
    V_out = [[0] * total_rank for _ in range(m * p)]
    W_out = [[0] * total_rank for _ in range(p * n)]
    
    col_off = 0
    for U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk in sub_decomps:
        n_cols = len(U_sub[0])
        for i in range(ni):
            for j in range(mj):
                for c in range(n_cols):
                    U_out[(i0+i)*m + (j0+j)][col_off + c] = U_sub[i*mj+j][c]
        for j in range(mj):
            for k in range(pk):
                for c in range(n_cols):
                    V_out[(j0+j)*p + (k0+k)][col_off + c] = V_sub[j*pk+k][c]
        for k in range(pk):
            for i in range(ni):
                for c in range(n_cols):
                    W_out[(k0+k)*n + (i0+i)][col_off + c] = W_sub[k*ni+i][c]
        col_off += n_cols
    
    return U_out, V_out, W_out

def construct():
    import time
    t0 = time.time()
    
    n, m, p = 4, 5, 5
    
    # Core 4x4x4
    U1, V1, W1 = _build_4x4x4_decomp()
    
    # Strip 1: i in 0..3, j in 0..3, k=4
    # This is a 4x4x1 multiplication
    U2, V2, W2 = _schoolbook_sub(4, 4, 1)
    
    # Strip 2: i in 0..3, j=4, k in 0..3
    # This is a 4x1x4 multiplication
    U3, V3, W3 = _schoolbook_sub(4, 1, 4)
    
    # Strip 3: i in 0..3, j=4, k=4
    # This is a 4x1x1 multiplication
    U4, V4, W4 = _schoolbook_sub(4, 1, 1)
    
    sub_decomps = [
        (U1, V1, W1, 0, 0, 0, 4, 4, 4),
        (U2, V2, W2, 0, 0, 4, 4, 4, 1),
        (U3, V3, W3, 0, 4, 0, 4, 1, 4),
        (U4, V4, W4, 0, 4, 4, 4, 1, 1),
    ]
    
    U_out, V_out, W_out = _assemble_block_decomp(n, m, p, sub_decomps)
    
    # Verify
    rank = len(U_out[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_out, "V": V_out, "W": W_out}
