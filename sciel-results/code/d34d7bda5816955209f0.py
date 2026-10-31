import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    # Build the decomposition by recursively applying Strassen.
    # We represent the tensor as a list of terms, each term being
    # (a_indices, b_indices, c_indices) where each is a dict from index to coeff.
    # Actually, let me think more carefully.
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # A decomposition of rank r gives U, V, W where:
    # U: shape (16, r) -- indices (i*m+j) for i,j in {0,1,2,3}, m=4
    # V: shape (16, r) -- indices (j*p+k) for j,k in {0,1,2,3}, p=4  
    # W: shape (16, r) -- indices (k*n+i) for k,i in {0,1,2,3}, n=4
    # with T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For Strassen on 2x2:
    # C = A*B where A,B are 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we split into 2x2 blocks:
    # A = [[A11, A12],[A21, A22]] where each Ai is 2x2
    # B = [[B11, B12],[B21, B22]] where each Bj is 2x2
    # C = [[C11, C12],[C21, C22]] where each Ci is 2x2
    # C11 = M1+M4-M5+M7 where Mk are 2x2 products
    # Each Mk = Xk * Yk where Xk, Yk are linear combinations of 2x2 blocks
    
    # So we need to decompose each 2x2 product Xk*Yk using Strassen 2x2.
    # That gives 7 scalar multiplications per block product.
    # Total: 7 (outer) * 7 (inner) = 49 terms.
    
    # Let me build this systematically.
    # 
    # Global indices: A has indices (row, col) in 0..3, so flat index = row*4+col
    # B has indices (row, col) in 0..3, flat index = row*4+col
    # C has indices (row, col) in 0..3, flat index = row*4+col
    # But W is indexed by (k*n+i) where n=4, so W index = col*4+row (since output C[row,col] = sum_k A[row,k]*B[k,col])
    # Wait, let me re-read: W[k*n+i,r] with n=4, so W index = k*4+i where i is row, k is col of C.
    # Actually the convention says W[k*n+i,r] and the tensor is T[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k]
    # So U[i*m+j, r] with m=4: U index = i*4+j (row of A, col of A)
    # V[j*p+k, r] with p=4: V index = j*4+k (row of B, col of B)
    # W[k*n+i, r] with n=4: W index = k*4+i (col of C, row of C)
    
    # Let me just build the 49 terms directly.
    
    # Each term is a triple (u_dict, v_dict, w_dict) where each dict maps flat index -> coeff
    # Then U[flat_idx, term] = u_dict[flat_idx], etc.
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Helper: create a dict from list of (key, val)
    def mkdict(pairs):
        d = {}
        for k, v in pairs:
            d[k] = d.get(k, 0) + v
        return d
    
    # Strassen 2x2 inner decomposition for a product of two 2x2 matrices
    # The 2x2 matrices are indexed locally:
    # Local A: (0,0)->flat_a00, (0,1)->flat_a01, (1,0)->flat_a10, (1,1)->flat_a11
    # Local B: (0,0)->flat_b00, (0,1)->flat_b01, (1,0)->flat_b10, (1,1)->flat_b11
    # Local C: (0,0)->flat_c00, (0,1)->flat_c01, (1,0)->flat_c10, (1,1)->flat_c11
    # 
    # But we need to map local indices to global flat indices.
    # For a 2x2 block at global position (br, bc) (block row, block col), 
    # the local (lr, lc) maps to global row = br*2+lr, global col = bc*2+lc
    # A flat index = global_row * 4 + global_col
    # B flat index = global_row * 4 + global_col
    # C: W index = col*4+row = global_col*4+global_row
    
    # The 7 Strassen terms for 2x2:
    # S1: A:(0,0)+(1,1)  B:(0,0)+(1,1)  C:(0,0)+(1,1)
    # S2: A:(1,0)+(1,1)  B:(0,0)         C:(1,0)-(1,1)  [wait, let me recheck]
    
    # Standard Strassen 2x2:
    # M1 = (A11+A22)(B11+B22): C11 += M1, C22 += M1
    # M2 = (A21+A22)(B11):     C21 += M2, C22 -= M2
    # M3 = (A11)(B12-B22):     C12 += M3, C22 += M3
    # M4 = (A22)(B21-B11):     C11 += M4, C21 += M4
    # M5 = (A11+A12)(B22):     C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12): C22 += M6
    # M7 = (A12-A22)(B21+B22): C11 += M7
    
    # So for each M_s, the contribution to C is:
    # S1: C00+=1, C11+=1
    # S2: C10+=1, C11+=-1
    # S3: C01+=1, C11+=1
    # S4: C00+=1, C10+=1
    # S5: C00+=-1, C01+=1
    # S6: C11+=1
    # S7: C00+=1
    
    strassen_2x2 = [
        # (a_comb, b_comb, c_comb)
        # a_comb: list of (local_row, local_col, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # S1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # S2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # S3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # S4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # S5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # S6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # S7
    ]
    
    # Outer Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00):     C10 += M2, C11 -= M2
    # M3 = (A00)(B01-B11):     C01 += M3, C11 += M3
    # M4 = (A11)(B10-B00):     C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11):     C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01): C11 += M6
    # M7 = (A01-A11)(B10+B11): C00 += M7
    
    # Block positions: A00 is at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
    # Each block is 2x2, so local (lr,lc) in block (br,bc) maps to global (br*2+lr, bc*2+lc)
    
    outer_strassen = [
        # (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (block_row, block_col, coeff)
        # b_blocks: list of (block_row, block_col, coeff)
        # c_blocks: list of (block_row, block_col, coeff)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # M2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # M5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # M6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # M7
    ]
    
    # Now for each outer term (a_blocks, b_blocks, c_blocks),
    # and for each inner Strassen term (a_comb, b_comb, c_comb),
    # we create a rank-1 term.
    
    # For the outer term, A_block = sum of global A blocks with coefficients.
    # The inner Strassen operates on the 2x2 entries of these blocks.
    
    # Let me think about this more carefully.
    # Outer M_s = X_s * Y_s where X_s is a 2x2 matrix (linear combo of A blocks) and Y_s is a 2x2 matrix (linear combo of B blocks).
    # Then M_s contributes to C blocks via the c_blocks coefficients.
    # 
    # X_s[local_r, local_c] = sum_{(br,bc,coeff) in a_blocks} coeff * A_global[br*2+local_r, bc*2+local_c]
    # Similarly for Y_s.
    #
    # Then inner Strassen: M_s = sum_{t=1}^{7} X_s * a_t(local) * Y_s * b_t(local) * c_t contribution
    # Wait, no. The inner Strassen decomposes the 2x2 product X_s * Y_s into 7 scalar multiplications.
    # Each scalar multiplication is: (linear combo of entries of X_s) * (linear combo of entries of Y_s)
    # and contributes to specific entries of the result 2x2 matrix.
    
    # So for inner term t:
    # scalar = (sum_{(lr,lc,ca) in a_comb} ca * X_s[lr,lc]) * (sum_{(lr,lc,cb) in b_comb} cb * Y_s[lr,lc])
    # And this scalar contributes to C_s[lr,lc] += cc * scalar for each (lr,lc,cc) in c_comb
    
    # Now substitute X_s[lr,lc] = sum_{(br,bc,coeff) in a_blocks} coeff * A_global[br*2+lr, bc*2+lc]
    # So the U part: for each (lr,lc,ca) in a_comb and (br,bc,coeff) in a_blocks:
    #   U index = (br*2+lr)*4 + (bc*2+lc), coeff = ca * coeff
    # Similarly for V: for each (lr,lc,cb) in b_comb and (br,bc,coeff) in b_blocks:
    #   V index = (br*2+lr)*4 + (bc*2+lc), coeff = cb * coeff
    # And for W: for each (lr,lc,cc) in c_comb and (br,bc,coeff) in c_blocks:
    #   W index = (bc*2+lc)*4 + (br*2+lr)  [since W[k*n+i] = col*4+row, and global col=bc*2+lc, row=br*2+lr]
    #   coeff = cc * coeff
    
    # Wait, I need to double-check the W indexing.
    # W[k*n+i, r] where n=4. The output C[i,k] = sum_j A[i,j]*B[j,k].
    # So for the decomposition: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = C[i,k]
    # So W index = k*4+i = col*4+row of C.
    # For a C block at (br, bc) with local position (lr, lc):
    # global row = br*2+lr, global col = bc*2+lc
    # W index = global_col * 4 + global_row = (bc*2+lc)*4 + (br*2+lr)
    
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_strassen):
        for inner_idx, (a_comb, b_comb, c_comb) in enumerate(strassen_2x2):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Build U contributions
            for (lr, lc, ca) in a_comb:
                for (br, bc, coeff) in a_blocks:
                    gr = br * 2 + lr  # global row
                    gc = bc * 2 + lc  # global col
                    u_idx = gr * 4 + gc
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
            
            # Build V contributions
            for (lr, lc, cb) in b_comb:
                for (br, bc, coeff) in b_blocks:
                    gr = br * 2 + lr  # global row
                    gc = bc * 2 + lc  # global col
                    v_idx = gr * 4 + gc
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
            
            # Build W contributions
            for (lr, lc, cc) in c_comb:
                for (br, bc, coeff) in c_blocks:
                    gr = br * 2 + lr  # global row of C
                    gc = bc * 2 + lc  # global col of C
                    w_idx = gc * 4 + gr  # W index = col*4+row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
            
            add_term(u_dict, v_dict, w_dict)
    
    rank = len(terms)
    
    # Verify rank is 49
    # 7 outer * 7 inner = 49
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_dict.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_dict.items():
            if val != 0:
                W[idx][col] = val
    
    # Verify the decomposition
    # C[i,k] should equal sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] summed over j
    # Let me verify with numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    C_computed = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for k in range(4):
            s = 0.0
            for j in range(4):
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*4+i, r]
            C_computed[i, k] = s
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # So C[i,k] = sum_j A[i,j]*B[j,k] where A=B=identity should give identity
    # But actually, the tensor decomposition should satisfy:
    # For any A, B: C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # In terms of the decomposition: C[i,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_k V[j*4+k,r]*B[j,k]) * W[k*4+i,r]
    # Wait, that's not quite right. Let me re-derive.
    
    # The standard form: C[i,k] = sum_r U[i*4+*, r] ... no.
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Hmm, that doesn't look right either because j is summed twice.
    
    # Let me use the standard tensor decomposition form:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k in {0,1,2,3}.
    
    # Let me verify this directly.
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*4+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
    else:
        # Convert to integer/half-integer if needed (should already be integers)
        pass
    
    return {"U": U, "V": V, "W": W}
